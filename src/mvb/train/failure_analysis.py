"""Failure analysis: turn a set of grades into the next dataset.

The loop this closes is the whole method. An accuracy number tells you a run was
worse; it does not tell you what to label next. These functions answer that
question directly:

* `failure_table` -- which mode dominates, and on which task.
* `worst_slices` -- which tag intersections are underperforming the mean by
  enough to be worth targeted collection.
* `mine_hard_negatives` -- the specific examples to oversample next round.
* `provenance_gap` -- whether a provenance class is inflating val relative to
  test, which is the check that killed the synthetic-augmented pool in v3.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from ..schemas import Example, FailureMode, Grade, Split


@dataclass
class SliceGap:
    name: str
    n: int
    accuracy: float
    gap: float
    """Accuracy minus the overall mean. Negative is bad."""

    dominant_failure: str


def failure_table(examples: list[Example], grades: list[Grade]) -> dict[str, dict[str, int]]:
    """Failure counts as task x mode. The cell that matters is usually obvious."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex, g in zip(examples, grades, strict=True):
        if g.correct:
            continue
        table[ex.task.value][g.failure_mode.value] += 1
    return {k: dict(sorted(v.items(), key=lambda p: -p[1])) for k, v in sorted(table.items())}


def worst_slices(
    examples: list[Example], grades: list[Grade], min_n: int = 25, top: int = 8
) -> list[SliceGap]:
    """Rank tag combinations by how far below the mean they sit.

    Single tags and pairs are both considered, because the interesting cell is
    almost always an intersection -- `occluded` alone was fine; `occluded` plus
    `crowded` was 19 points below the mean and is what v3 was trained to fix.
    """
    overall = sum(1 for g in grades if g.correct) / max(1, len(grades))

    buckets: dict[str, list[tuple[Example, Grade]]] = defaultdict(list)
    for ex, g in zip(examples, grades, strict=True):
        keys = ["clean"] if not ex.tags else list(ex.tags)
        if len(ex.tags) > 1:
            keys.append("+".join(sorted(ex.tags)))
        for k in keys:
            buckets[k].append((ex, g))

    out: list[SliceGap] = []
    for name, rows in buckets.items():
        if len(rows) < min_n:
            continue
        acc = sum(1 for _, g in rows if g.correct) / len(rows)
        fails = Counter(g.failure_mode.value for _, g in rows if not g.correct)
        out.append(
            SliceGap(
                name=name,
                n=len(rows),
                accuracy=acc,
                gap=acc - overall,
                dominant_failure=fails.most_common(1)[0][0] if fails else "none",
            )
        )
    out.sort(key=lambda s: s.gap)
    return out[:top]


def mine_hard_negatives(
    examples: list[Example],
    grades: list[Grade],
    modes: set[FailureMode] | None = None,
    limit: int = 200,
) -> list[Example]:
    """Select examples to oversample in the next training round.

    Sorted by difficulty *ascending*, which is counter-intuitive and deliberate:
    an example the model got wrong despite being easy is a cleaner training
    signal than one it got wrong because it is genuinely ambiguous. The
    adversarial tail is mostly label noise and teaches the model to hedge.
    """
    modes = modes or {
        FailureMode.SPATIAL_INVERSION,
        FailureMode.HALLUCINATED_OBJECT,
        FailureMode.MISCOUNT,
        FailureMode.TEMPORAL_CONFUSION,
    }
    picked = [
        ex
        for ex, g in zip(examples, grades, strict=True)
        if not g.correct and g.failure_mode in modes and ex.split is Split.TRAIN
    ]
    picked.sort(key=lambda e: (e.difficulty, e.id))
    return picked[:limit]


def provenance_gap(examples: list[Example], grades: list[Grade]) -> dict[str, dict[str, float]]:
    """Per-provenance accuracy split by val vs test.

    A provenance class that scores much better on val than on test is being
    memorised, not learned. That divergence is invisible in the headline number
    and is the single most useful check in this module.
    """
    buckets: dict[tuple[str, str], list[Grade]] = defaultdict(list)
    for ex, g in zip(examples, grades, strict=True):
        if ex.split is Split.TRAIN:
            continue
        buckets[(ex.provenance.value, ex.split.value)].append(g)

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (prov, split), rows in buckets.items():
        out[prov][split] = round(sum(1 for g in rows if g.correct) / len(rows), 4)
        out[prov][f"{split}_n"] = len(rows)
    for row in out.values():
        if "val" in row and "test" in row:
            row["val_minus_test"] = round(row["val"] - row["test"], 4)
    return dict(out)


def recommendations(examples: list[Example], grades: list[Grade]) -> list[str]:
    """Turn the analysis into instructions a human can act on this week."""
    out: list[str] = []
    table = failure_table(examples, grades)

    for task, modes in table.items():
        if not modes:
            continue
        mode, count = next(iter(modes.items()))
        total = sum(modes.values())
        if total and count / total > 0.4:
            if mode == FailureMode.FORMAT_VIOLATION.value:
                out.append(
                    f"{task}: {count}/{total} failures are format violations. "
                    "This is an SFT problem, not a data-volume problem -- train on the output "
                    "shape before collecting more examples."
                )
            elif mode == FailureMode.SPATIAL_INVERSION.value:
                out.append(
                    f"{task}: {count}/{total} failures are inverted relations. "
                    "Collect paired examples that differ only in the relation term."
                )
            else:
                out.append(f"{task}: dominated by {mode} ({count}/{total}). Target that mode directly.")

    for s in worst_slices(examples, grades)[:3]:
        if s.gap < -0.05:
            out.append(
                f"slice {s.name!r} is {abs(s.gap) * 100:.1f} points below the mean over {s.n} "
                f"examples, mostly {s.dominant_failure}. Mine hard negatives here."
            )

    for prov, row in provenance_gap(examples, grades).items():
        delta = row.get("val_minus_test")
        if delta is not None and delta > 0.05:
            out.append(
                f"provenance {prov!r} scores {delta * 100:.1f} points higher on val than test. "
                "It is being memorised -- consider dropping it or re-splitting."
            )

    return out
