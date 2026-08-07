#!/usr/bin/env python3
"""The rehearsed walkthrough. Six beats, no live network, ~40 seconds."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvb.evalsuite.suite import load_suite, run_suite  # noqa: E402
from mvb.observability.trace import get_tracer  # noqa: E402
from mvb.serve.agent import run_agent  # noqa: E402
from mvb.tools import registry  # noqa: E402
from mvb.train.failure_analysis import recommendations, worst_slices  # noqa: E402


def beat(n: int, title: str) -> None:
    print(f"\n\033[1m[{n}] {title}\033[0m\n" + "-" * 70)


def main() -> int:
    suite = Path("eval/datasets/suite.jsonl")
    if not suite.exists():
        print("run `make data` first")
        return 1
    examples = load_suite(suite)

    beat(1, "The suite: 1,500 examples, five tasks, five slices")
    from collections import Counter

    print("  tasks:  ", dict(Counter(e.task.value for e in examples)))
    print("  slices: ", dict(Counter(t for e in examples for t in (e.tags or ["clean"]))))

    beat(2, "Base checkpoint vs post-trained adapter")
    sample = examples[:400]
    base, _, _ = run_suite(sample, adapter="base")
    tuned, _, grades = run_suite(sample, adapter="lora-v3")
    print(f"  {base.summary_line()}")
    print(f"  {tuned.summary_line()}")

    beat(3, "Where the base model actually failed")
    for mode, count in list(base.failure_counts.items())[:4]:
        print(f"  {mode:<24}{count}")
    print("\n  Format violations dominate -- that is an SFT problem, not a data-volume problem.")

    beat(4, "What to build next, derived from the grades")
    for s in worst_slices(sample, grades, min_n=15, top=3):
        print(f"  {s.name:<24} acc={s.accuracy:.3f} gap={s.gap:+.3f}  ({s.dominant_failure})")
    for line in recommendations(sample, grades)[:3]:
        print(f"  - {line}")

    beat(5, "The same model behind a tool-calling agent")
    print(f"  tools: {', '.join(registry.names())}")
    run = run_agent("which cameras saw a blocked keep-clear aisle")
    print(f"  Q: {run.question}")
    print(f"  A: {run.answer}")
    print(f"  tools used: {run.tool_names()}")

    beat(6, "Every step of that was traced")
    for name, stats in get_tracer().summary().items():
        print(f"  {name:<20}n={stats['count']:<4}p50={stats['p50_ms']:>7.2f}ms  errors={stats['errors']}")

    print("\nSame tools are exposed over MCP: `mvb mcp --print-config`\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
