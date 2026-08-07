#!/usr/bin/env python3
"""Generate the raw pool, curate it, and write the shipped suite."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvb.data.curate import curate  # noqa: E402
from mvb.data.synth import generate_pool  # noqa: E402
from mvb.evalsuite.suite import write_suite  # noqa: E402
from mvb.schemas import Provenance, Split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1900, help="raw pool size before curation")
    ap.add_argument("--size", type=int, default=1500, help="exact size of the shipped suite")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="eval/datasets")
    ap.add_argument(
        "--drop-synthetic",
        action="store_true",
        help="reproduce the v2->v3 change: exclude synthetic_augmented examples",
    )
    args = ap.parse_args()

    pool = generate_pool(n=args.n, seed=args.seed)
    exclude = {Provenance.SYNTHETIC_AUGMENTED} if args.drop_synthetic else set()
    rows, report = curate(
        pool, exclude_provenance=exclude, seed=args.seed, target_size=args.size
    )

    out = Path(args.out)
    n_all = write_suite(out / "suite.jsonl", rows)
    for split in Split:
        write_suite(out / f"{split.value}.jsonl", [r for r in rows if r.split is split])

    print("\n".join(report.lines()))
    print()
    for split in Split:
        k = sum(1 for r in rows if r.split is split)
        print(f"{split.value:<6} {k}")
    print(f"\nwrote {n_all} examples to {out}/suite.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
