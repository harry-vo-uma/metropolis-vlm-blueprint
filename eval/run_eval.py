#!/usr/bin/env python3
"""Run every adapter over the suite and print the comparison in the README.

This script is the source of truth for the reported numbers. If a figure appears
in the documentation and cannot be produced by running this file, that figure is
a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvb.evalsuite.suite import load_suite, regressions, run_suite, save_report  # noqa: E402
from mvb.train.failure_analysis import (  # noqa: E402
    failure_table,
    provenance_gap,
    recommendations,
    worst_slices,
)

ADAPTERS = ["base", "sft-v1", "lora-v2", "lora-v3"]


def rule(char: str = "-", n: int = 78) -> str:
    return char * n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="eval/datasets/suite.jsonl")
    ap.add_argument("--adapters", nargs="*", default=ADAPTERS)
    ap.add_argument("--out", default="eval/reports")
    ap.add_argument("--json", action="store_true", help="dump machine-readable results and exit")
    args = ap.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"suite not found at {suite_path}. Run `make data` first.", file=sys.stderr)
        return 1

    examples = load_suite(suite_path)
    results = {}
    graded = {}

    for adapter in args.adapters:
        report, _preds, grades = run_suite(examples, adapter=adapter)
        results[adapter] = report
        graded[adapter] = grades
        save_report(report, Path(args.out) / f"{adapter}.json")

    if args.json:
        print(
            json.dumps(
                {a: r.model_dump() for a, r in results.items()},
                indent=2,
            )
        )
        return 0

    print(f"\nsuite: {suite_path}  n={len(examples)}\n")

    # -- headline -----------------------------------------------------------
    print(rule("="))
    print(f"{'adapter':<12}{'accuracy':>10}{'mean score':>12}{'p50 ms':>9}{'p95 ms':>9}")
    print(rule())
    for adapter, r in results.items():
        print(
            f"{adapter:<12}{r.accuracy:>10.3f}{r.mean_score:>12.3f}"
            f"{r.p50_latency_ms:>9.0f}{r.p95_latency_ms:>9.0f}"
        )
    print(rule("="))

    first, last = args.adapters[0], args.adapters[-1]
    delta = results[last].accuracy - results[first].accuracy
    print(
        f"\n{first} -> {last}: {results[first].accuracy:.1%} -> {results[last].accuracy:.1%} "
        f"({delta:+.1%} absolute, {delta / max(1e-9, results[first].accuracy):+.1%} relative)\n"
    )

    # -- per task -----------------------------------------------------------
    tasks = sorted(results[first].by_task)
    print("accuracy by task")
    print(rule())
    header = f"{'task':<24}" + "".join(f"{a:>12}" for a in args.adapters)
    print(header)
    for task in tasks:
        row = f"{task:<24}"
        for a in args.adapters:
            row += f"{results[a].by_task[task].accuracy:>12.3f}"
        print(row)
    print()

    # -- per slice ----------------------------------------------------------
    tags = sorted(results[first].by_tag)
    print("accuracy by slice")
    print(rule())
    print(f"{'slice':<24}" + "".join(f"{a:>12}" for a in args.adapters))
    for tag in tags:
        row = f"{tag:<24}"
        for a in args.adapters:
            s = results[a].by_tag.get(tag)
            row += f"{s.accuracy:>12.3f}" if s else f"{'-':>12}"
        print(row)
    print()

    # -- failures -----------------------------------------------------------
    print("failure modes")
    print(rule())
    modes = sorted({m for r in results.values() for m in r.failure_counts})
    print(f"{'mode':<24}" + "".join(f"{a:>12}" for a in args.adapters))
    for mode in modes:
        row = f"{mode:<24}"
        for a in args.adapters:
            row += f"{results[a].failure_counts.get(mode, 0):>12}"
        print(row)
    print()

    # -- regressions --------------------------------------------------------
    regressed = list(regressions(examples, graded[first], graded[last]))
    print(f"regressions ({first} correct, {last} wrong): {len(regressed)}")
    for ex, _before, after in regressed[:5]:
        print(f"  [{ex.task.value}/{after.failure_mode.value}] {ex.prompt[:64]}")
    print()

    # -- failure table for the final adapter --------------------------------
    print(f"failure table for {last}")
    print(rule())
    for task, modes_ in failure_table(examples, graded[last]).items():
        print(f"  {task:<24}{modes_}")
    print()

    print(f"worst slices for {last}")
    print(rule())
    for s in worst_slices(examples, graded[last]):
        print(f"  {s.name:<28}n={s.n:<5}acc={s.accuracy:.3f}  gap={s.gap:+.3f}  {s.dominant_failure}")
    print()

    print(f"provenance val/test gap for {last}")
    print(rule())
    for prov, row in sorted(provenance_gap(examples, graded[last]).items()):
        print(f"  {prov:<24}{row}")
    print()

    print("recommended next actions")
    print(rule())
    for line in recommendations(examples, graded[last]):
        print(f"  - {line}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
