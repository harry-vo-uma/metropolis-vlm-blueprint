import json

from conftest import make_example

from mvb.evalsuite.graders import extract_json, grade, grade_spatial, token_f1
from mvb.schemas import FailureMode, Prediction, TaskKind


def _pred(ex, raw):
    return Prediction(example_id=ex.id, task=ex.task, raw=raw)


def test_clean_json_is_distinguished_from_fenced_json():
    """The fence is the format signal; a lenient parser destroys it."""
    obj = {"count": 2}
    clean, was_clean = extract_json(json.dumps(obj))
    assert (clean, was_clean) == (obj, True)

    fenced, was_clean = extract_json(f"Sure!\n```json\n{json.dumps(obj)}\n```")
    assert fenced == obj and was_clean is False


def test_correct_json_in_prose_still_fails_and_is_named_a_format_violation():
    ex = make_example(task=TaskKind.ATTRIBUTE_EXTRACTION, target=json.dumps({"count": 2}))
    g = grade(ex, _pred(ex, f"Here you go:\n```json\n{ex.target}\n```"))
    assert not g.correct
    assert g.failure_mode is FailureMode.FORMAT_VIOLATION


def test_inverted_spatial_relation_scores_zero_despite_high_token_overlap():
    """Token F1 rates this ~0.9. That is the bug grade_spatial exists to fix."""
    ref = "The blue forklift is left of the pallet."
    pred = "The blue forklift is right of the pallet."
    assert token_f1(pred, ref) > 0.7  # a fully inverted answer, rated as near-correct
    assert grade_spatial(pred, ref) == 0.0


def test_spatial_inversion_is_attributed_not_lumped_into_other():
    ex = make_example(
        task=TaskKind.SPATIAL_RELATION,
        prompt="Where is the forklift?",
        target="The forklift is left of the pallet.",
    )
    g = grade(ex, _pred(ex, "The forklift is right of the pallet."))
    assert g.failure_mode is FailureMode.SPATIAL_INVERSION


def test_far_from_is_not_read_as_near():
    """Longest-match matters: 'far from' contains no 'near', but 'near' is a
    substring of nothing here -- the reverse case is the trap."""
    assert grade_spatial("The cart is far from the dock.", "The cart is near the dock.") == 0.0


def test_miscount_cannot_pass_on_token_overlap():
    ex = make_example(target="3 forklifts are queued at the dock.")
    g = grade(ex, _pred(ex, "4 forklifts are queued at the dock."))
    assert not g.correct
    assert g.failure_mode is FailureMode.MISCOUNT


def test_reordered_temporal_answer_is_temporal_confusion_not_generic_failure():
    ex = make_example(
        task=TaskKind.TEMPORAL_ORDERING,
        prompt="Order them.",
        target=json.dumps(["a", "b", "c"]),
    )
    g = grade(ex, _pred(ex, json.dumps(["b", "a", "c"])))
    assert not g.correct
    assert g.failure_mode is FailureMode.TEMPORAL_CONFUSION


def test_verdict_without_justification_fails_the_judgement_task():
    ex = make_example(task=TaskKind.ANOMALY_JUDGEMENT, target="Yes", rationale="the aisle is blocked")
    bare = grade(ex, _pred(ex, "Yes"))
    full = grade(ex, _pred(ex, "Yes. Because the aisle is blocked by a stationary pallet"))
    assert not bare.correct
    assert full.correct


def test_refusal_is_attributed_to_overcautious_refusal():
    ex = make_example()
    g = grade(ex, _pred(ex, "I cannot determine that from the image provided."))
    assert g.failure_mode is FailureMode.OVERCAUTIOUS_REFUSAL
