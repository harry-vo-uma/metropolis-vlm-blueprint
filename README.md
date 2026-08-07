# Metropolis VLM Blueprint

A reference architecture for post-training, evaluating, and serving
vision-language models on NVIDIA NIM — with the part most blueprints skip: an
evaluation loop that tells you what to fix, not just how well you did.

Runs offline on a laptop. No GPU, no API key, no network. Set `NVIDIA_API_KEY`
and the same code runs against live NIM with no changes.

```bash
git clone https://github.com/harry-vo-uma/metropolis-vlm-blueprint
cd metropolis-vlm-blueprint && make install
mvb doctor && make data && make report
```

---

## What it does

**Post-training pipeline.** Dataset synthesis, curation with a printed audit
trail, LoRA/SFT configuration, prompt-masked label construction, and run
manifests. Four adapters, each one a response to a specific finding in the
previous run's failure table.

**A 1,500-example evaluation suite.** Five tasks, five visual condition slices
and their intersections, four provenance classes, hash-assigned splits. Rule-based
graders that are honest about their limits.

**Failure attribution.** Eight failure modes, attributed per example, rolled up
by task and slice, ending in imperative next actions.

**Serving.** FastAPI inference, a tool registry that validates itself at import,
an agent loop with real guardrails, an MCP server, and nested tracing — one
process, one port.

## Results

`make data && make report` produces all of this. Nothing below is hard-coded.

```
adapter       accuracy  mean score   p50 ms   p95 ms
------------------------------------------------------------------------------
base             0.701       0.802      145      190
sft-v1           0.791       0.847      134      164
lora-v2          0.829       0.883      134      163
lora-v3          0.878       0.918      133      164
```

**70.1% → 87.8%** across four adapters (+17.7 points absolute, +25.3% relative).

| task | base | sft-v1 | lora-v2 | lora-v3 |
|---|---|---|---|---|
| anomaly_judgement | 0.798 | 0.798 | 0.826 | 0.876 |
| attribute_extraction | 0.515 | 0.778 | 0.813 | 0.867 |
| scene_qa | 0.938 | 0.935 | 0.938 | 0.946 |
| spatial_relation | 0.658 | 0.705 | 0.801 | 0.876 |
| temporal_ordering | 0.556 | 0.677 | 0.707 | 0.778 |

The shape of that table is the argument for slicing. `scene_qa` is flat at ~0.94
throughout and contributes nothing to the gain; at 369 examples it would have
diluted the headline enough to make the whole effort look marginal. All the
movement is in the three tasks that were actually broken.

| failure mode | base | sft-v1 | lora-v2 | lora-v3 |
|---|---|---|---|---|
| format_violation | 144 | 25 | 16 | 8 |
| spatial_inversion | 105 | 93 | 64 | 40 |
| temporal_confusion | 53 | 44 | 44 | 37 |
| hallucinated_object | 47 | 43 | 33 | 36 |
| miscount | 45 | 48 | **58** | 27 |

Read the columns, not the total. `sft-v1` cut format violations 83% and moved
almost nothing else — exactly what a format-only pass should do, which is
evidence the attribution is real. `lora-v2` traded miscounts *upward* while
cutting spatial inversions; `lora-v3` is the first adapter to improve both.

## Two design decisions worth arguing about

**Scoring and attribution are separate, deliberately.** The score answers "did we
ship a regression". The attribution answers "what do we do about it". Collapsing
them into a single number is how teams end up with a dashboard that goes down
and no idea which dataset to go build. `make report` ends in sentences like:

```
spatial_relation: 40/40 failures are inverted relations.
  Collect paired examples that differ only in the relation term.
slice 'motion_blur+night' is 35.9 points below the mean over 27 examples,
  mostly spatial_inversion. Mine hard negatives here.
provenance 'rule_derived' scores 12.1 points higher on val than test.
  It is being memorised -- consider dropping it or re-splitting.
```

`motion_blur+night` at 0.519 is the clearest finding in the run, and no
single-tag analysis surfaces it — neither tag alone is below 0.80. `worst_slices`
considers pairwise intersections for exactly that reason.

**Fixing the graders lowered the numbers, and that was correct.** Two real bugs
surfaced during development, both cases where the harness reported success on
answers that were plainly wrong:

- Token F1 rated a *fully inverted* spatial relation at ~0.9 — "left of the
  pallet" and "right of the pallet" differ in one token out of eight, so the
  whole `SPATIAL_RELATION` column read as solved. `grade_spatial` now scores a
  wrong relation term as exactly 0.0.
- A changed digit barely moved token F1. In this domain the number usually *is*
  the answer, so free-form scores are capped at 0.30 on numeric disagreement.

Base accuracy fell from 0.747 to 0.671. A benchmark that will not go down is not
measuring anything. A third fix — the rationale requirement in the judgement
task was decorative, since a bare "Yes" scored 0.70 against a 0.62 bar — is in
`docs/evaluation.md`.

## Developer experience

Fifteen external testers were given only the repo URL and a task. Unassisted
first-run success went **57% → 89%**, median time to a first prediction ~6 hours
→ ~45 minutes. The largest single contributor was not a code change; it was
`mvb doctor`, because five of six first-round failures were environmental and
were being diagnosed by reading tracebacks.

The full write-up, including six prioritized product recommendations across
model selection, tool registration, evaluation, and deployment friction — and
three requests that were deliberately declined — is in
[`docs/product-feedback.md`](docs/product-feedback.md).

## Commands

```bash
mvb doctor                      # environment preflight
mvb data --n 1900 --size 1500   # generate and curate the suite
mvb eval --by-task              # score adapters
mvb tools                       # list registered tools
mvb tools --call count_objects --arguments '{"camera_id": "aisle-03"}'
mvb ask "which cameras saw a blocked keep-clear aisle"
mvb serve                       # FastAPI on :8000
mvb mcp --print-config          # MCP client config stanza
make demo                       # six-beat walkthrough, ~40s
```

## About the default backend

**It is not a model.** `nim/mock.py` is a behavioural simulator: given an
adapter, task, and example tags it computes a competence probability, then
either emits a correct answer or corrupts one in a task-appropriate way — a real
code fence for `FORMAT_VIOLATION`, an actual `left`/`right` swap for
`SPATIAL_INVERSION`. The graders have to detect those from the string, which is
how the grader bugs above were found.

What that buys you: a deterministic, free, offline CI signal, and a harness
exercised against every failure mode before it meets a real model. What it does
not buy you: any evidence about a real model's capability. `/api/health` reports
the backend actually in use, because the most common question during testing was
"is it really hitting the model" and the honest answer was often no.

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Clone to served endpoint |
| [Architecture](docs/architecture.md) | Five layers, and what is deliberately absent |
| [Post-training](docs/post-training.md) | Why each adapter exists |
| [Evaluation](docs/evaluation.md) | Curation, grading, slices, regressions |
| [API examples](docs/api-examples.md) | Every endpoint, request and response |
| [Troubleshooting](docs/troubleshooting.md) | Ordered by frequency |
| [Product feedback](docs/product-feedback.md) | Six prioritized recommendations |
| [Demo script](docs/demo-script.md) | Narration for `make demo` |

`templates/` has four runnable starting points: a minimal prediction, a custom
tool, a custom grader, and evaluating your own data.

## Companion repository

[**agentic-warehouse-vision**](https://github.com/harry-vo-uma/agentic-warehouse-vision)
— the multi-camera DeepStream + VLM pipeline and agent graph this blueprint's
models are built for.

## License

Apache-2.0.
