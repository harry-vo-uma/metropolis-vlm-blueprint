"""The evaluation harness.

Loads a JSONL suite, runs an adapter over it, grades every prediction, and slices
the result three ways: by task, by tag, and by provenance. The provenance slice
exists because it is the one that caught a real problem -- see
`docs/post-training.md`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..nim.client import get_backend
from ..schemas import EvalReport, Example, Grade, Prediction, SliceResult, Split
from .graders import grade


def load_suite(path: str | Path, split: Split | None = None) -> list[Example]:
    rows: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = Example.model_validate_json(line)
            if split is None or ex.split is split:
                rows.append(ex)
    return rows


def write_suite(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
            n += 1
    return n


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def _slice(name: str, grades: list[Grade]) -> SliceResult:
    n = len(grades)
    if n == 0:
        return SliceResult(name=name, n=0, accuracy=0.0, mean_score=0.0)
    return SliceResult(
        name=name,
        n=n,
        accuracy=sum(1 for g in grades if g.correct) / n,
        mean_score=sum(g.score for g in grades) / n,
    )


def run_suite(
    examples: list[Example],
    adapter: str = "base",
    backend=None,
) -> tuple[EvalReport, list[Prediction], list[Grade]]:
    backend = backend or get_backend()
    preds: list[Prediction] = []
    grades: list[Grade] = []

    for ex in examples:
        pred = backend.predict(ex, adapter=adapter)
        preds.append(pred)
        grades.append(grade(ex, pred))

    by_task: dict[str, list[Grade]] = defaultdict(list)
    by_tag: dict[str, list[Grade]] = defaultdict(list)
    by_prov: dict[str, list[Grade]] = defaultdict(list)

    for ex, g in zip(examples, grades, strict=True):
        by_task[ex.task.value].append(g)
        by_prov[ex.provenance.value].append(g)
        for tag in ex.tags:
            by_tag[tag].append(g)
        if not ex.tags:
            by_tag["clean"].append(g)

    latencies = [p.latency_ms for p in preds]
    n = len(grades) or 1

    report = EvalReport(
        adapter=adapter,
        n=len(grades),
        accuracy=sum(1 for g in grades if g.correct) / n,
        mean_score=sum(g.score for g in grades) / n,
        by_task={k: _slice(k, v) for k, v in sorted(by_task.items())},
        by_tag={k: _slice(k, v) for k, v in sorted(by_tag.items())},
        by_provenance={k: _slice(k, v) for k, v in sorted(by_prov.items())},
        failure_counts=dict(
            Counter(g.failure_mode.value for g in grades if not g.correct).most_common()
        ),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )
    return report, preds, grades


def regressions(
    examples: list[Example], before: list[Grade], after: list[Grade]
) -> Iterator[tuple[Example, Grade, Grade]]:
    """Examples the newer adapter got wrong that the older one got right.

    Aggregate accuracy hides these completely. Every post-training round in this
    repo produced some, and `docs/post-training.md` reports the count rather than
    pretending the improvement was free.
    """
    index = {g.example_id: g for g in before}
    for ex, g_after in zip(examples, after, strict=True):
        g_before = index.get(ex.id)
        if g_before is not None and g_before.correct and not g_after.correct:
            yield ex, g_before, g_after


def save_report(report: EvalReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
