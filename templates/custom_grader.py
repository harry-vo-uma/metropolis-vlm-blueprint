"""Grade a task your own way.

Graders are plain functions over (Example, Prediction). Swapping one in is a
local change -- nothing else in the harness needs to know.

Run: python templates/custom_grader.py
"""

from mvb.evalsuite.graders import token_f1
from mvb.schemas import Example, Prediction, Provenance, Split, TaskKind


def strict_containment(ex: Example, pred: Prediction) -> float:
    """Require every content word of the reference to appear in the prediction.

    Harsher than token F1 and much less forgiving of paraphrase -- which is the
    right trade when the downstream consumer is a rule, not a human reader.
    """
    ref_terms = {t for t in ex.target.lower().split() if len(t) > 3}
    got = pred.raw.lower()
    return sum(1 for t in ref_terms if t in got) / max(1, len(ref_terms))


if __name__ == "__main__":
    ex = Example(
        id="g1",
        task=TaskKind.SCENE_QA,
        split=Split.TEST,
        provenance=Provenance.HUMAN_LABELLED,
        prompt="What is at the dock?",
        target="A yellow forklift is unloading pallets.",
    )
    pred = Prediction(example_id="g1", task=ex.task, raw="A forklift unloads pallets at the dock.")
    print("token_f1           ", round(token_f1(pred.raw, ex.target), 3))
    print("strict_containment ", round(strict_containment(ex, pred), 3))
