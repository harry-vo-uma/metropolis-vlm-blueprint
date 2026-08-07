"""Point the harness at your own JSONL.

Your file needs one JSON object per line matching mvb.schemas.Example. The
loader validates it, so a malformed record fails immediately with the field name
rather than producing a quietly wrong score.

Run: python templates/eval_your_data.py eval/datasets/test.jsonl
"""

import sys

from mvb.evalsuite.suite import load_suite, run_suite
from mvb.train.failure_analysis import recommendations, worst_slices

path = sys.argv[1] if len(sys.argv) > 1 else "eval/datasets/test.jsonl"
examples = load_suite(path)
report, _preds, grades = run_suite(examples, adapter="lora-v3")

print(report.summary_line())
print("\nworst slices:")
for s in worst_slices(examples, grades, min_n=10, top=5):
    print(f"  {s.name:<24} n={s.n:<4} acc={s.accuracy:.3f} gap={s.gap:+.3f}")
print("\nnext actions:")
for line in recommendations(examples, grades):
    print(f"  - {line}")
