"""Smallest useful thing: one image, one question, one answer.

Run: python templates/minimal_predict.py
Works with no API key -- it falls back to the mock backend.
"""

from mvb.nim.client import get_backend
from mvb.schemas import Example, Frame, Provenance, Split, TaskKind

ex = Example(
    id="demo-1",
    task=TaskKind.SCENE_QA,
    split=Split.TEST,
    provenance=Provenance.HUMAN_LABELLED,
    frames=[Frame(uri="frames/dock-a/00001_0.jpg", camera="dock-a")],
    prompt="Is the loading dock obstructed?",
    target="n/a",  # no reference: we are predicting, not grading
)

pred = get_backend().predict(ex, adapter="lora-v3")
print(pred.raw)
print(f"({pred.latency_ms:.0f} ms, {pred.completion_tokens} completion tokens)")
