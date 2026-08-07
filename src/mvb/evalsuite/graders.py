"""Grading and failure attribution.

Two jobs, deliberately kept separate:

1. **Score** the prediction against the reference.
2. **Attribute** a failure mode when the score is low.

Keeping them separate matters because the score answers "did we ship a
regression" and the attribution answers "what do we do about it". Collapsing
them into a single number is how teams end up with a dashboard that goes down
and no idea which dataset to go build.

The graders are all rule-based and local. Using a strong LLM as a judge would
score better on free-form tasks, but it would also mean the evaluation loop
costs money, is non-deterministic, and can regress underneath you when the judge
model is updated. The tradeoff -- and the resulting soft ceiling on SCENE_QA
grading fidelity -- is documented in `docs/evaluation.md`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import get_settings
from ..schemas import Example, FailureMode, Grade, Prediction, TaskKind

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "and", "or", "it", "its", "this", "that", "there", "with", "for", "be",
}

_REFUSAL_MARKERS = (
    "cannot determine",
    "not enough",
    "unable to",
    "i can't",
    "i cannot",
    "insufficient information",
    "no way to tell",
)

_HALLUCINATION_MARKERS = ("there is also", "extra_object", "additionally, i can see")

_SPATIAL_TERMS = (
    "left", "right", "above", "below", "in front of", "behind", "inside",
    "outside", "near", "far from",
)

_OPPOSITES = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "in front of": "behind",
    "behind": "in front of",
    "inside": "outside",
    "outside": "inside",
    "near": "far from",
    "far from": "near",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1. Not a great metric; it is an *honest* one.

    It rewards saying the right things and penalises padding, which is exactly
    the pressure we want on a model whose output feeds an alerting UI. It cannot
    tell a paraphrase from a contradiction, and `docs/evaluation.md` says so.
    """
    p, r = _tokens(pred), _tokens(ref)
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_json(raw: str) -> tuple[Any | None, bool]:
    """Try to parse JSON out of a model response.

    Returns `(value, was_clean)`. `was_clean` is False when the JSON was only
    recoverable after stripping prose or code fences -- that distinction is the
    entire FORMAT_VIOLATION signal, so it must not be swallowed by a lenient
    parser. A lot of eval harnesses strip fences unconditionally and then report
    that their model has no formatting problem.
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped), True
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip()), False
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1]), False
            except json.JSONDecodeError:
                continue
    return None, False


def _json_similarity(pred: Any, ref: Any) -> float:
    """Field-level agreement. Extra keys are penalised, not ignored."""
    if isinstance(ref, dict) and isinstance(pred, dict):
        keys = set(ref) | set(pred)
        if not keys:
            return 1.0
        hits = sum(1 for k in keys if k in ref and k in pred and pred[k] == ref[k])
        return hits / len(keys)
    if isinstance(ref, list) and isinstance(pred, list):
        if not ref:
            return 1.0 if not pred else 0.0
        # Positional agreement, because for TEMPORAL_ORDERING the order *is* the
        # answer. Set overlap would score a reversed sequence as perfect.
        matched = sum(1 for a, b in zip(pred, ref, strict=False) if a == b)
        return matched / max(len(ref), len(pred))
    return 1.0 if pred == ref else 0.0


def _relation_in(text: str) -> str | None:
    """Longest matching spatial term, so `far from` is not read as `near`'s absence."""
    low = text.lower()
    hits = [t for t in _SPATIAL_TERMS if t in low]
    return max(hits, key=len) if hits else None


def grade_spatial(pred: str, ref: str) -> float:
    """Spatial answers are graded on the relation term, not on token overlap.

    Token F1 is actively misleading here. "The blue forklift is left of the
    pallet" and "...is right of the pallet" differ in one token out of eight, so
    F1 scores a fully inverted answer around 0.9 and the whole SPATIAL_RELATION
    column reads as solved. The relation *is* the answer; everything else in the
    sentence is scaffolding copied from the question.
    """
    ref_rel = _relation_in(ref)
    if ref_rel is None:
        return token_f1(pred, ref)

    pred_rel = _relation_in(pred)
    if pred_rel is None:
        return 0.0
    if pred_rel != ref_rel:
        return 0.0
    # Relation is right; award partial credit for naming the correct referents.
    return 0.7 + 0.3 * token_f1(pred, ref)


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def _attribute(ex: Example, raw: str, score: float, clean_json: bool | None) -> FailureMode:
    """Assign a failure mode. Order matters: the cheapest, most certain checks
    first, so a format violation is never mislabelled as a content error."""
    low = raw.lower()

    if ex.is_structured() and clean_json is False:
        return FailureMode.FORMAT_VIOLATION
    if ex.task is TaskKind.TEMPORAL_ORDERING and clean_json is None:
        return FailureMode.FORMAT_VIOLATION

    if any(m in low for m in _REFUSAL_MARKERS):
        return FailureMode.OVERCAUTIOUS_REFUSAL
    if any(m in low for m in _HALLUCINATION_MARKERS):
        return FailureMode.HALLUCINATED_OBJECT

    if ex.task is TaskKind.SPATIAL_RELATION:
        ref_low = ex.target.lower()
        for term in _SPATIAL_TERMS:
            if term in ref_low and _OPPOSITES[term] in low and term not in low:
                return FailureMode.SPATIAL_INVERSION

    if ex.task is TaskKind.TEMPORAL_ORDERING:
        pred, _ = extract_json(raw)
        ref = ex.parsed_target()
        if isinstance(pred, list) and isinstance(ref, list) and sorted(map(str, pred)) == sorted(
            map(str, ref)
        ):
            # Same events, wrong order. This is the defining temporal error and
            # would otherwise be indistinguishable from a generic low score.
            return FailureMode.TEMPORAL_CONFUSION

    ref_nums = re.findall(r"\d+", ex.target)
    pred_nums = re.findall(r"\d+", raw)
    if ref_nums and pred_nums and ref_nums != pred_nums:
        return FailureMode.MISCOUNT

    return FailureMode.OTHER


def grade(ex: Example, pred: Prediction) -> Grade:
    cfg = get_settings().grading
    raw = pred.raw
    clean_json: bool | None = None

    if ex.is_structured():
        parsed, clean = extract_json(raw)
        clean_json = clean if parsed is not None else None
        if parsed is None:
            score = 0.0
        else:
            score = _json_similarity(parsed, ex.parsed_target())
            if not clean:
                # A correct object smuggled inside prose still breaks the caller.
                # Capping below the strict threshold is the point: it fails, and
                # it fails for a reason the report can name.
                score = min(score, 0.5)
        threshold = cfg.structured_threshold

    elif ex.task is TaskKind.ANOMALY_JUDGEMENT:
        verdict_ref = ex.target.strip().lower()
        head = raw.strip().lower()
        verdict_ok = head.startswith(verdict_ref) or verdict_ref in head.split(".")[0]
        has_reason = len(raw.split()) >= 8 and (" because" in raw.lower() or ";" in raw)
        # The verdict alone must land *below* the pass threshold, otherwise the
        # rationale requirement is decorative: a bare "Yes" would score 0.7 and
        # sail past a 0.62 bar, and the whole point of this task is that an
        # alerting product cannot act on an unjustified verdict.
        score = (0.6 if verdict_ok else 0.0) + (0.4 if has_reason else 0.0)
        if not cfg.judgement_requires_rationale and verdict_ok:
            score = 1.0
        threshold = cfg.freeform_threshold

    elif ex.task is TaskKind.SPATIAL_RELATION:
        score = grade_spatial(raw, ex.target)
        threshold = cfg.freeform_threshold

    else:
        score = token_f1(raw, ex.target)
        ref_nums, pred_nums = _numbers(ex.target), _numbers(raw)
        if ref_nums and pred_nums != ref_nums:
            # In this domain the number usually *is* the answer -- "3 forklifts
            # are queued" and "4 forklifts are queued" are different facts with
            # a token F1 around 0.9. Cap hard so a miscount cannot pass.
            score = min(score, 0.30)
        threshold = cfg.freeform_threshold

    correct = score >= threshold
    mode = FailureMode.CORRECT if correct else _attribute(ex, raw, score, clean_json)

    detail = ""
    if not correct:
        detail = f"score={score:.3f} < {threshold:.3f}"

    return Grade(
        example_id=ex.id,
        task=ex.task,
        correct=correct,
        score=score,
        failure_mode=mode,
        detail=detail,
        tags=list(ex.tags),
    )
