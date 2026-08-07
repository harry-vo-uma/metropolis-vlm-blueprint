# Post-training

Four adapters, each one a response to a specific finding in the previous run's
failure table. The progression matters more than the endpoint: the point of this
document is that every training decision was traceable to an attributed failure
mode rather than to "more data, more steps".

| adapter | change | accuracy | what it was answering |
|---|---|---|---|
| `base` | none | 0.701 | — |
| `sft-v1` | format-only SFT | 0.791 | 144 format violations |
| `lora-v2` | + spatial data, rationale supervision | 0.829 | 105 spatial inversions, unjustified verdicts |
| `lora-v3` | + hard-negative mining, drop synthetic | 0.878 | occluded/crowded gap, val-test divergence |

## base → sft-v1: fix the shape before fixing the content

The base failure table was dominated by one cell: 144 of 448 failures were
`format_violation`, and 118 of those were `attribute_extraction`. The model
knew the attributes. It wrapped them in a code fence and an apology, and the
caller broke.

This is an SFT problem, not a data-volume problem, and `recommendations()` says
so in exactly those words when the cell exceeds 40% of a task's failures:

> `attribute_extraction: 118/163 failures are format violations. This is an SFT
> problem, not a data-volume problem -- train on the output shape before
> collecting more examples.`

So `sft-v1` trains only on output shape — same content, correct envelope. Format
violations fell 144 → 25 (−83%). `attribute_extraction` went 0.515 → 0.778. Every
other task moved less than two points, `scene_qa` moved −0.003.

That flatness is the useful part. A run that improves one attributed mode and
leaves the others alone is evidence the attribution is real. A run that improves
everything by 5% is evidence you changed something you do not understand.

## sft-v1 → lora-v2: the relation is the answer

With format handled, the largest remaining cell was `spatial_inversion` at 93.
`recommendations()` again:

> `spatial_relation: 93/93 failures are inverted relations. Collect paired
> examples that differ only in the relation term.`

Paired examples are the specific fix. If the training data contains "the pallet
is left of the dock" but never the near-identical "…right of the dock", the model
can satisfy the loss by learning the *scene* and guessing the relation. Pairs
that differ in exactly one token make the relation the only thing that
distinguishes them, and force it into the gradient.

`lora-v2` also adds rationale supervision, driven by the judgement task. Setting
`include_rationale=True` in `format_example` appends the justification to the
target so the model learns verdict-plus-reason as one unit rather than learning
that a bare verdict is acceptable. This pairs with the grader change described in
`docs/evaluation.md` — the 0.6/0.4 split that puts an unjustified verdict below
threshold. Changing the grader without changing the training data would have been
moving the goalposts; changing both is closing the loop.

Result: spatial_relation 0.705 → 0.801, anomaly_judgement 0.798 → 0.826.

`lora-v2` also made things worse: `miscount` went 48 → 58. Adding spatial data
without rebalancing counting data cost counting accuracy. This is the ordinary
case, not an anomaly, and it is why the failure table is printed per-adapter
rather than only for the best one.

## lora-v2 → lora-v3: mine the intersections, drop what is memorised

Two findings drove v3.

**The worst cell was an intersection.** `occluded` alone was fine; `occluded` plus
`crowded` sat well below the mean. `worst_slices` considers single tags *and*
pairs for exactly this reason — the interesting cell is almost always an
intersection, and single-tag analysis will report that both tags are acceptable
while the region where they overlap is 20 points down.

`mine_hard_negatives` then selects the examples to oversample. It sorts by
difficulty **ascending**, which is counter-intuitive and deliberate:

> an example the model got wrong despite being easy is a cleaner training signal
> than one it got wrong because it is genuinely ambiguous. The adversarial tail
> is mostly label noise and teaches the model to hedge.

Oversampling the hardest examples is the standard instinct and it produces a
model that refuses more. `overcautious_refusal` is one of the eight tracked modes
precisely so that this is visible when it happens.

**The synthetic-augmented pool was being memorised.** `provenance_gap` showed it
scoring materially higher on val than on test. v3 was trained with
`--drop-synthetic`. The headline barely moved; the val-test gap closed, which
means the *reported* number became trustworthy even though it did not become
larger.

Result: 0.829 → 0.878, with `occluded` 0.773 → 0.862 and `crowded` 0.809 → 0.890.

## Mechanics

LoRA, rank 16, alpha 32, dropout 0.05. Target modules are the attention
projections plus the multimodal connector:

```python
["q_proj", "k_proj", "v_proj", "o_proj"] + (["mm_projector"] if train_connector else [])
```

Including `mm_projector` is what makes this *visual* adaptation rather than a
language-side style transfer. If the failures are spatial and you only tune
attention on the text side, you are teaching the model to describe what it
already saw slightly differently.

Labels are prompt-masked with `IGNORE_INDEX = -100`, so loss is computed on the
completion only. Without masking, the model spends most of its capacity learning
to reproduce the instruction template.

Every run writes a `RunManifest`: base model, adapter name, LoRA config, dataset
fingerprint, example count, seed, timestamp. An adapter you cannot trace back to
a specific dataset revision is an adapter you cannot debug.

## Running it

The training path needs the extra:

```bash
pip install -e ".[train]"
python -m mvb.train.sft --adapter lora-v4 --suite eval/datasets/train.jsonl
```

Everything up to the point where a GPU is genuinely required — dataset
formatting, label masking, LoRA config construction, manifest building — is
importable and testable without torch. `train()` lazily imports torch and peft
and exits with an actionable message if the extra is missing, rather than
failing at module import and making the whole package unusable on a laptop.

## What is not claimed here

The adapter numbers in this repo are produced against the mock backend described
in `docs/architecture.md`. They demonstrate that the *pipeline* — curation,
grading, attribution, slice analysis, hard-negative mining, regression detection
— behaves correctly and produces actionable output. They are not evidence about
any real model's capability. Set `NVIDIA_API_KEY` and rerun before quoting a
number anywhere it matters.
