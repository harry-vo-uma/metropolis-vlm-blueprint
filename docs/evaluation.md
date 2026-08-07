# Evaluation

The suite is 1,500 examples across five tasks. Every number in this document is
produced by `make data && make report`; if a figure appears here and cannot be
reproduced by running those two commands, that figure is a bug.

## Suite composition

```
input                 1900
dropped_duplicate       47
dropped_balance        275
dropped_trim            78
kept                  1500

train 1037 | val 167 | test 296
```

| task | n |
|---|---|
| attribute_extraction | 369 |
| scene_qa | 369 |
| spatial_relation | 322 |
| anomaly_judgement | 242 |
| temporal_ordering | 198 |

Tag slices: `night`, `occluded`, `small_object`, `crowded`, `motion_blur`, and
their pairwise intersections. Untagged examples are bucketed as `clean`.

## Curation, stage by stage

**Deduplicate** on a fingerprint of `(task, normalised prompt, normalised
target)`. In an early version of the generator this stage removed 878 of 2,100
examples and collapsed `anomaly_judgement` to eight unique rows. The dedup count
was the only thing that revealed the generators' combinatorial space was far too
small — a lesson worth keeping the counter for.

**Balance** caps any task at 1.35× the median task count. Left alone the
generator produces 38% `scene_qa`, which is also the task every adapter is
already good at; an unbalanced suite would let a model coast to a high headline
number without improving on anything hard.

**Drop provenance** is optional (`--drop-synthetic`) and exists because of the
val-vs-test check described below.

**Trim to size** removes the easiest example from the currently largest task
bucket, repeatedly, until the suite is exactly the requested size. Trimming from
the easy end rather than randomly keeps the difficulty distribution from
drifting each time the target size changes.

## Splits are assigned by hash, not by shuffle

```python
def split_examples(examples, val_frac=0.12, test_frac=0.20, seed=1337):
    """Assign splits by *fingerprint hash*, not by shuffling."""
```

An example lands in the same split every time, regardless of how the pool grows
around it. Shuffle-based splitting means that adding 200 examples next month
re-deals the entire deck and quietly moves last month's test examples into
train. The model then scores beautifully on data it was trained on and nobody
can explain why the production numbers do not match.

## Grading

Rule-based and local. A strong LLM judge would score free-form answers better,
but it would make the evaluation loop cost money, become non-deterministic, and
be able to regress underneath you when the judge model is updated. The cost of
that tradeoff is a soft ceiling on `scene_qa` grading fidelity, and it is real:
token F1 cannot tell a paraphrase from a contradiction. `scene_qa` accuracy
should be read as a floor.

Per task:

| task | method | pass threshold |
|---|---|---|
| attribute_extraction, temporal_ordering | JSON field/positional agreement | 0.999 |
| spatial_relation | relation-term match, then token F1 for referents | 0.62 |
| anomaly_judgement | verdict (0.6) + rationale present (0.4) | 0.62 |
| scene_qa | token F1, capped at 0.30 on numeric disagreement | 0.62 |

### Two grader bugs worth documenting

Both were found by running the mock's deliberately-corrupted outputs through the
harness and noticing that the harness said they were fine.

**Token F1 rated inverted spatial relations at ~0.9.** "The blue forklift is left
of the pallet" and "…is right of the pallet" differ in one token out of eight.
The whole `SPATIAL_RELATION` column read as solved. The relation *is* the
answer; everything else in the sentence is scaffolding copied from the question.
`grade_spatial` now scores a wrong relation term as exactly 0.0, and awards
`0.7 + 0.3 * token_f1` when it is right.

**A changed digit barely moved token F1.** "3 forklifts are queued" versus "4
forklifts are queued" scores around 0.9. In this domain the number usually *is*
the answer, so free-form scores are now capped at 0.30 when the digits disagree
with the reference.

Fixing both *lowered* the headline numbers — base fell from 0.747 to 0.671 — which
is the correct direction. A benchmark that will not go down is not measuring
anything.

**A third: the rationale requirement was decorative.** Judgement scoring was
0.7 verdict / 0.3 rationale, so a bare "Yes" scored 0.70 and sailed past the 0.62
bar. An alerting product cannot act on an unjustified verdict. The split is now
0.6 / 0.4 so a verdict alone lands below threshold by construction.

### Format violations are a first-class signal

`extract_json` returns `(value, was_clean)`. `was_clean` is False when the JSON
was only recoverable after stripping prose or code fences. Many harnesses strip
fences unconditionally and then report that their model has no formatting
problem. Here, a correct object smuggled inside prose is capped at 0.5 — below
the structured threshold — so it fails, and it fails for a reason the report can
name.

## Results

```
adapter       accuracy  mean score   p50 ms   p95 ms
------------------------------------------------------------------------------
base             0.701       0.802      145      190
sft-v1           0.791       0.847      134      164
lora-v2          0.829       0.883      134      163
lora-v3          0.878       0.918      133      164
```

base → lora-v3: **70.1% → 87.8%** (+17.7 points absolute, +25.3% relative).

### By task

| task | base | sft-v1 | lora-v2 | lora-v3 |
|---|---|---|---|---|
| anomaly_judgement | 0.798 | 0.798 | 0.826 | 0.876 |
| attribute_extraction | 0.515 | 0.778 | 0.813 | 0.867 |
| scene_qa | 0.938 | 0.935 | 0.938 | 0.946 |
| spatial_relation | 0.658 | 0.705 | 0.801 | 0.876 |
| temporal_ordering | 0.556 | 0.677 | 0.707 | 0.778 |

The shape of this table is the argument for slicing. `scene_qa` is flat at ~0.94
throughout — it contributes nothing to the gain and, at 369 examples, would have
diluted the headline enough to make the whole effort look marginal. All of the
movement is in `attribute_extraction` (+35 points, almost entirely format),
`spatial_relation` (+22), and `temporal_ordering` (+22).

### By slice

| slice | base | lora-v3 |
|---|---|---|
| clean | 0.767 | 0.900 |
| crowded | 0.659 | 0.890 |
| occluded | 0.607 | 0.862 |
| small_object | 0.649 | 0.864 |
| night | 0.631 | 0.818 |
| motion_blur | 0.627 | 0.802 |

### Failure modes

| mode | base | sft-v1 | lora-v2 | lora-v3 |
|---|---|---|---|---|
| format_violation | 144 | 25 | 16 | 8 |
| spatial_inversion | 105 | 93 | 64 | 40 |
| temporal_confusion | 53 | 44 | 44 | 37 |
| hallucinated_object | 47 | 43 | 33 | 36 |
| miscount | 45 | 48 | 58 | 27 |
| overcautious_refusal | 31 | 41 | 25 | 27 |
| other | 24 | 20 | 16 | 8 |

Read the columns, not the total. `sft-v1` cut format violations by 83% and moved
almost nothing else — which is exactly what a format-only SFT pass should do,
and confirms the attribution is working. `lora-v2` traded miscounts *upward*
(45 → 58) while cutting spatial inversions; `lora-v3` is the first adapter to
improve both.

## Regressions

108 examples were correct under `base` and wrong under `lora-v3`. That is 7.2%
of the suite moving backwards inside a run that gained 17.7 points, and it is
invisible in any aggregate. `regressions()` in `evalsuite/suite.py` lists them;
`make report` prints the first five with their attributed failure mode. Ship
review should look at this list, not just the delta.

## The provenance check

Per-provenance accuracy, split val versus test, for `lora-v3`:

| provenance | val | test | val − test |
|---|---|---|---|
| rule_derived | 0.933 | 0.813 | **+0.121** |
| human_labelled | 0.857 | 0.847 | +0.010 |
| synthetic_augmented | 0.844 | 0.913 | −0.069 |
| model_distilled | 0.870 | 0.951 | −0.082 |

A provenance class that scores much better on val than on test is being
memorised, not learned. `rule_derived` is 12 points higher on val, which is the
signal that its templates are narrow enough for the model to pattern-match. This
divergence is completely invisible in the headline number and is the single most
useful check in the module.

## What the analysis recommends

`make report` ends by turning the above into instructions:

```
- spatial_relation: 40/40 failures are inverted relations.
  Collect paired examples that differ only in the relation term.
- attribute_extraction: dominated by hallucinated_object (27/49).
- temporal_ordering: dominated by temporal_confusion (37/44).
- slice 'motion_blur+night' is 35.9 points below the mean over 27 examples,
  mostly spatial_inversion. Mine hard negatives here.
- provenance 'rule_derived' scores 12.1 points higher on val than test.
  It is being memorised -- consider dropping it or re-splitting.
```

`motion_blur+night` at 0.519 is the clearest finding in the run: neither tag
alone is below 0.80, but their intersection is 36 points under the mean. That
cell is what the next collection round should target, and no single-tag analysis
would have surfaced it.
