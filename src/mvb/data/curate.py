"""Dataset curation.

The unglamorous half of post-training. Three operations, in this order:

1. **Deduplicate** on the (prompt, target) pair. Near-duplicates in a multimodal
   set are common because augmentation pipelines re-emit the same question
   against slightly different crops, and they inflate val accuracy without
   teaching the model anything.
2. **Balance** across tasks, so the loss is not dominated by whichever task was
   easiest to label. Left unbalanced, SCENE_QA was 61% of the raw pool.
3. **Order** by difficulty into a curriculum, because on a rank-16 adapter with
   three epochs, showing the adversarial tail first measurably destabilised the
   early steps.

Every operation returns a report rather than silently mutating, so the numbers in
`docs/post-training.md` can be regenerated instead of remembered.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..schemas import Example, Provenance, Split, TaskKind


@dataclass
class CurationReport:
    input_n: int = 0
    output_n: int = 0
    dropped_duplicates: int = 0
    dropped_balance: int = 0
    dropped_provenance: int = 0
    dropped_trim: int = 0
    per_task: dict[str, int] = field(default_factory=dict)
    per_provenance: dict[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [
            f"input:                {self.input_n}",
            f"dropped duplicates:   {self.dropped_duplicates}",
            f"dropped for balance:  {self.dropped_balance}",
            f"dropped provenance:   {self.dropped_provenance}",
            f"dropped to hit size:  {self.dropped_trim}",
            f"output:               {self.output_n}",
            "",
            "per task:",
        ]
        out += [f"  {k:<24} {v}" for k, v in sorted(self.per_task.items())]
        out += ["", "per provenance:"]
        out += [f"  {k:<24} {v}" for k, v in sorted(self.per_provenance.items())]
        return out


def _fingerprint(ex: Example) -> str:
    """Normalised (prompt, target) hash.

    Case and whitespace are stripped because the augmentation pass that produced
    a good chunk of the pool varied capitalisation and nothing else.
    """
    key = f"{ex.task.value}|{' '.join(ex.prompt.lower().split())}|{' '.join(ex.target.lower().split())}"
    return hashlib.sha1(key.encode()).hexdigest()


def deduplicate(examples: list[Example]) -> tuple[list[Example], int]:
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in examples:
        fp = _fingerprint(ex)
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(ex)
    return kept, len(examples) - len(kept)


def balance_tasks(
    examples: list[Example], cap_ratio: float = 1.35, seed: int = 1337
) -> tuple[list[Example], int]:
    """Cap any task at `cap_ratio` times the median task count.

    A hard equal-count cap throws away too much of the tasks that are genuinely
    easier to collect. Capping relative to the median keeps the natural skew
    while removing the pathological one.
    """
    buckets: dict[TaskKind, list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[ex.task].append(ex)
    if not buckets:
        return [], 0

    counts = sorted(len(v) for v in buckets.values())
    median = counts[len(counts) // 2]
    cap = max(1, int(median * cap_ratio))

    rng = random.Random(seed)
    kept: list[Example] = []
    for task in sorted(buckets, key=lambda t: t.value):
        rows = buckets[task]
        if len(rows) > cap:
            # Keep the hardest examples preferentially -- easy ones are the ones
            # the model already gets right, so they contribute least gradient.
            rows = sorted(rows, key=lambda e: -e.difficulty)[: cap + len(rows) // 10]
            rng.shuffle(rows)
            rows = rows[:cap]
        kept.extend(rows)
    return kept, len(examples) - len(kept)


def drop_provenance(
    examples: list[Example], exclude: set[Provenance]
) -> tuple[list[Example], int]:
    """Remove whole provenance classes.

    This exists because of a specific finding: `SYNTHETIC_AUGMENTED` examples
    lifted val accuracy by 3.1 points and moved held-out test accuracy by 0.2.
    They were teaching the model the augmentation artefacts. Dropping them is the
    v2 -> v3 change.
    """
    if not exclude:
        return list(examples), 0
    kept = [ex for ex in examples if ex.provenance not in exclude]
    return kept, len(examples) - len(kept)


def trim_to_size(examples: list[Example], target: int) -> tuple[list[Example], int]:
    """Trim to an exact size by repeatedly dropping the easiest example from the
    currently largest task bucket.

    A fixed suite size is worth a little effort: it makes accuracy numbers
    comparable across regenerations, and it means "1,500-example suite" is a
    fact about the artefact rather than an approximation. Dropping easy examples
    from the biggest bucket keeps the balance achieved above rather than undoing
    it, and dropping easy ones costs the least signal.

    Drops are tracked by *position*, not by example id. Ids are content hashes,
    so two identical rows share one -- and an earlier version that collected
    dropped ids into a set removed every row sharing a dropped id, overshooting
    the target whenever the input had not already been deduplicated. `curate`
    always dedups first, which is precisely why the bug survived there and only
    appeared when the function was called on its own.
    """
    if target <= 0 or len(examples) <= target:
        return list(examples), 0

    buckets: dict[TaskKind, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        buckets[ex.task].append(i)
    for idxs in buckets.values():
        # Ties broken on id, then position, so the result does not depend on
        # input ordering and is stable when ids collide.
        idxs.sort(key=lambda i: (examples[i].difficulty, examples[i].id, i))

    to_drop = len(examples) - target
    dropped: set[int] = set()
    for _ in range(to_drop):
        task = max(buckets, key=lambda t: (len(buckets[t]), t.value))
        dropped.add(buckets[task].pop(0))

    kept = [ex for i, ex in enumerate(examples) if i not in dropped]
    return kept, len(examples) - len(kept)


def curriculum_order(examples: list[Example], seed: int = 1337) -> list[Example]:
    """Sort into easy -> hard, with local shuffling inside difficulty bands.

    Pure difficulty sorting correlates difficulty with training step, which lets
    the optimiser fit the schedule instead of the data. Banding then shuffling
    within a band keeps the coarse curriculum without that artefact.
    """
    rng = random.Random(seed)
    bands: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        bands[min(4, int(ex.difficulty * 5))].append(ex)
    ordered: list[Example] = []
    for band in sorted(bands):
        rows = bands[band]
        rng.shuffle(rows)
        ordered.extend(rows)
    return ordered


def split_examples(
    examples: list[Example], val_frac: float = 0.12, test_frac: float = 0.20, seed: int = 1337
) -> list[Example]:
    """Assign splits by *fingerprint hash*, not by shuffling.

    Hashing means an example lands in the same split every time regardless of
    how the pool grows, so adding data later cannot leak a previous test example
    into train. Shuffle-based splitting silently breaks this, and the breakage is
    invisible in the metrics -- it just makes them better.
    """
    out: list[Example] = []
    for ex in examples:
        bucket = int(_fingerprint(ex)[:8], 16) % 1000 / 1000.0
        if bucket < test_frac:
            split = Split.TEST
        elif bucket < test_frac + val_frac:
            split = Split.VAL
        else:
            split = Split.TRAIN
        out.append(ex.model_copy(update={"split": split}))
    return out


def curate(
    examples: list[Example],
    exclude_provenance: set[Provenance] | None = None,
    cap_ratio: float = 1.35,
    seed: int = 1337,
    target_size: int | None = None,
) -> tuple[list[Example], CurationReport]:
    report = CurationReport(input_n=len(examples))

    rows, report.dropped_duplicates = deduplicate(examples)
    rows, report.dropped_provenance = drop_provenance(rows, exclude_provenance or set())
    rows, report.dropped_balance = balance_tasks(rows, cap_ratio=cap_ratio, seed=seed)
    if target_size is not None:
        rows, report.dropped_trim = trim_to_size(rows, target_size)
    rows = split_examples(rows, seed=seed)
    rows = curriculum_order(rows, seed=seed)

    report.output_n = len(rows)
    report.per_task = dict(Counter(ex.task.value for ex in rows).most_common())
    report.per_provenance = dict(Counter(ex.provenance.value for ex in rows).most_common())
    return rows, report
