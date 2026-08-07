"""A deterministic stand-in for a hosted VLM.

This is not a model. It is a *behavioural simulator* whose only job is to make the
rest of the blueprint -- curation, training config, grading, failure analysis,
serving, tracing -- runnable and testable end to end on a laptop with no GPU and
no API key.

It matters that the simulator is wrong in the same *shapes* a real VLM is wrong,
not merely wrong at the same rate. A model that failed uniformly at random would
make the failure-analysis tooling look useful when it is not. So the error model
here is structured:

* competence varies by task, and the base checkpoint is much weaker on the two
  tasks that require a machine-readable answer;
* competence varies by slice -- `night`, `occluded`, `small_object` and `crowded`
  each carry their own penalty, and they stack;
* difficulty shifts the logit rather than the probability, so the hard tail is
  hard for every adapter;
* when the model fails, *which* way it fails is drawn from a per-task
  distribution, and the emitted text genuinely exhibits that failure. The grader
  is not told the failure mode -- it has to detect it from the string, exactly as
  it would with a real model.

The upshot is that `make eval` reproduces the reported base/post-trained gap by
actually running the grader over generated text, rather than by printing a
number that was hard-coded somewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any

from ..schemas import Example, FailureMode, Prediction, TaskKind

#: Per-adapter, per-task base competence expressed as a logit. Higher is better.
#: The base checkpoint is a generalist: acceptable at open-ended description,
#: poor at anything that must parse, and near-chance on relative spatial terms.
_TASK_LOGIT: dict[str, dict[TaskKind, float]] = {
    "base": {
        TaskKind.SCENE_QA: 2.11,
        TaskKind.ATTRIBUTE_EXTRACTION: 0.68,
        TaskKind.SPATIAL_RELATION: 0.56,
        TaskKind.ANOMALY_JUDGEMENT: 1.90,
        TaskKind.TEMPORAL_ORDERING: 0.51,
    },
    # First pass: format-focused SFT only. Fixes parsing, barely moves reasoning.
    # Kept in the repo because the flat spatial number is the reason v2 exists.
    "sft-v1": {
        TaskKind.SCENE_QA: 2.20,
        TaskKind.ATTRIBUTE_EXTRACTION: 1.85,
        TaskKind.SPATIAL_RELATION: 0.70,
        TaskKind.ANOMALY_JUDGEMENT: 1.70,
        TaskKind.TEMPORAL_ORDERING: 0.95,
    },
    # Adds spatial-relation data mined from the failure analysis, and rationale
    # supervision on the judgement task.
    "lora-v2": {
        TaskKind.SCENE_QA: 2.30,
        TaskKind.ATTRIBUTE_EXTRACTION: 2.05,
        TaskKind.SPATIAL_RELATION: 1.35,
        TaskKind.ANOMALY_JUDGEMENT: 1.85,
        TaskKind.TEMPORAL_ORDERING: 1.30,
    },
    # Adds hard-negative mining on the `occluded` and `crowded` slices, and drops
    # the synthetic-augmented examples that were inflating val without moving test.
    "lora-v3": {
        TaskKind.SCENE_QA: 2.32,
        TaskKind.ATTRIBUTE_EXTRACTION: 2.12,
        TaskKind.SPATIAL_RELATION: 1.62,
        TaskKind.ANOMALY_JUDGEMENT: 2.15,
        TaskKind.TEMPORAL_ORDERING: 1.42,
    },
}

#: Slice penalties in logit space. These stack, which is why the
#: `occluded`+`crowded` intersection is the worst cell in the report.
_TAG_PENALTY: dict[str, float] = {
    "night": 0.45,
    "occluded": 0.70,
    "small_object": 0.55,
    "crowded": 0.40,
    "motion_blur": 0.35,
}

#: How much of each slice penalty the adapter recovers. Hard-negative mining in
#: v3 targeted occlusion and crowding specifically, so those recover most.
_TAG_RECOVERY: dict[str, dict[str, float]] = {
    "base": {},
    "sft-v1": {"night": 0.10, "occluded": 0.05, "small_object": 0.05, "crowded": 0.05},
    "lora-v2": {"night": 0.35, "occluded": 0.30, "small_object": 0.30, "crowded": 0.30},
    "lora-v3": {"night": 0.50, "occluded": 0.72, "small_object": 0.45, "crowded": 0.70},
}

#: Failure-mode mixture per task, before adapter adjustment.
_FAILURE_MIX: dict[TaskKind, list[tuple[FailureMode, float]]] = {
    TaskKind.SCENE_QA: [
        (FailureMode.HALLUCINATED_OBJECT, 0.45),
        (FailureMode.MISCOUNT, 0.20),
        (FailureMode.OVERCAUTIOUS_REFUSAL, 0.15),
        (FailureMode.OTHER, 0.20),
    ],
    TaskKind.ATTRIBUTE_EXTRACTION: [
        (FailureMode.FORMAT_VIOLATION, 0.55),
        (FailureMode.MISCOUNT, 0.25),
        (FailureMode.HALLUCINATED_OBJECT, 0.20),
    ],
    TaskKind.SPATIAL_RELATION: [
        (FailureMode.SPATIAL_INVERSION, 0.65),
        (FailureMode.HALLUCINATED_OBJECT, 0.20),
        (FailureMode.OTHER, 0.15),
    ],
    TaskKind.ANOMALY_JUDGEMENT: [
        (FailureMode.OVERCAUTIOUS_REFUSAL, 0.35),
        (FailureMode.HALLUCINATED_OBJECT, 0.30),
        (FailureMode.OTHER, 0.35),
    ],
    TaskKind.TEMPORAL_ORDERING: [
        (FailureMode.TEMPORAL_CONFUSION, 0.55),
        (FailureMode.FORMAT_VIOLATION, 0.30),
        (FailureMode.OTHER, 0.15),
    ],
}

#: Format-following is the thing SFT fixes first and most completely. Applied as
#: a multiplier on the FORMAT_VIOLATION mass, with the remainder redistributed.
_FORMAT_FIX: dict[str, float] = {
    "base": 1.0,
    "sft-v1": 0.12,
    "lora-v2": 0.06,
    "lora-v3": 0.03,
}


def _seed(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def competence(adapter: str, ex: Example) -> float:
    """Probability this adapter answers this example correctly.

    Exposed rather than kept private because `scripts/sanity_check.py` sweeps it
    to confirm the simulated gap is a property of the profile and not of a lucky
    seed.
    """
    profile = _TASK_LOGIT.get(adapter, _TASK_LOGIT["base"])
    logit = profile[ex.task]

    recovery = _TAG_RECOVERY.get(adapter, {})
    for tag in ex.tags:
        penalty = _TAG_PENALTY.get(tag)
        if penalty is None:
            continue
        logit -= penalty * (1.0 - recovery.get(tag, 0.0))

    # Difficulty shifts the logit, so the adversarial tail stays hard for every
    # adapter instead of being trivially solved by a stronger one.
    logit -= 2.6 * (ex.difficulty - 0.5)
    return _sigmoid(logit)


def _pick_failure(adapter: str, ex: Example, rng: random.Random) -> FailureMode:
    mix = list(_FAILURE_MIX[ex.task])
    fix = _FORMAT_FIX.get(adapter, 1.0)

    adjusted: list[tuple[FailureMode, float]] = []
    freed = 0.0
    for mode, weight in mix:
        if mode is FailureMode.FORMAT_VIOLATION:
            kept = weight * fix
            freed += weight - kept
            adjusted.append((mode, kept))
        else:
            adjusted.append((mode, weight))

    if freed > 0 and len(adjusted) > 1:
        # Redistribute the mass SFT took off format onto the remaining modes, so
        # a post-trained model does not merely fail less -- it fails differently.
        others = [m for m in adjusted if m[0] is not FailureMode.FORMAT_VIOLATION]
        total = sum(w for _, w in others) or 1.0
        adjusted = [
            (m, w + freed * (w / total) if m is not FailureMode.FORMAT_VIOLATION else w)
            for m, w in adjusted
        ]

    modes = [m for m, _ in adjusted]
    weights = [w for _, w in adjusted]
    return rng.choices(modes, weights=weights, k=1)[0]


_SPATIAL_FLIP = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "in front of": "behind",
    "behind": "in front of",
    "inside": "outside",
    "outside": "inside",
    "near": "far from",
}

_REFUSALS = [
    "I cannot determine that from the image provided.",
    "There is not enough visual information to answer reliably.",
    "I'm unable to make that assessment from a single frame.",
]

CAMERA_HINTS = ["dock-a", "dock-b", "aisle-03", "aisle-07", "pack-01", "pack-03", "yard-n"]

_HALLUCINATED = ["a red toolbox", "a safety cone", "a second forklift", "an open ladder"]


def _corrupt(ex: Example, mode: FailureMode, rng: random.Random) -> str:
    """Produce text that genuinely exhibits `mode`.

    Deliberately does not tell the grader anything. If the grader cannot detect
    the mode from this string alone, that is a real gap in the grader.
    """
    target = ex.target

    if mode is FailureMode.FORMAT_VIOLATION:
        if ex.is_structured():
            # The classic: a correct answer wrapped in conversational prose and a
            # fenced block, which json.loads will not touch.
            return f"Sure! Here's what I found:\n\n```json\n{target}\n```\n\nLet me know if you need more detail."
        return target.upper() + " !!!"

    if mode is FailureMode.OVERCAUTIOUS_REFUSAL:
        return rng.choice(_REFUSALS)

    if mode is FailureMode.HALLUCINATED_OBJECT:
        ghost = rng.choice(_HALLUCINATED)
        if ex.is_structured():
            try:
                obj = json.loads(target)
                if isinstance(obj, dict):
                    obj["extra_object"] = ghost
                    return json.dumps(obj)
            except json.JSONDecodeError:
                pass
        return f"{target} There is also {ghost} visible in the scene."

    if mode is FailureMode.MISCOUNT:
        if ex.is_structured():
            try:
                obj = json.loads(target)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, int):
                            obj[k] = max(0, v + rng.choice([-2, -1, 1, 2]))
                            return json.dumps(obj)
            except json.JSONDecodeError:
                pass
        digits = [c for c in target if c.isdigit()]
        if digits:
            d = digits[0]
            return target.replace(d, str((int(d) + 1) % 10), 1)
        return target + " (approximately three of them)"

    if mode is FailureMode.SPATIAL_INVERSION:
        low = target.lower()
        for term, flipped in _SPATIAL_FLIP.items():
            if term in low:
                idx = low.index(term)
                return target[:idx] + flipped + target[idx + len(term) :]
        return "the opposite of " + target

    if mode is FailureMode.TEMPORAL_CONFUSION:
        try:
            obj = json.loads(target)
            if isinstance(obj, list) and len(obj) > 1:
                i = rng.randrange(len(obj) - 1)
                obj[i], obj[i + 1] = obj[i + 1], obj[i]
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        parts = target.split(", ")
        if len(parts) > 1:
            parts.reverse()
            return ", ".join(parts)
        return "after " + target

    words = target.split()
    if len(words) > 3:
        rng.shuffle(words)
        return " ".join(words)
    return "unclear"


class MockVLMBackend:
    """Deterministic backend. Same example + same adapter => same output, always.

    Determinism is not a nicety here: the whole evaluation story falls apart if a
    reported delta between two adapters could be sampling noise.
    """

    name = "mock"

    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed
        self.calls = 0

    def predict(self, ex: Example, adapter: str = "base") -> Prediction:
        self.calls += 1
        rng = random.Random(_seed(str(self.seed), ex.id, adapter))
        t0 = time.perf_counter()

        p = competence(adapter, ex)
        if rng.random() < p:
            raw = ex.target
            if ex.task is TaskKind.ANOMALY_JUDGEMENT:
                # The judgement task requires a justification; a bare verdict is
                # graded as incomplete even when the verdict itself is right.
                reason = ex.rationale or "the visual evidence in the frame supports this"
                raw = f"{ex.target}. Because {reason}"
        else:
            mode = _pick_failure(adapter, ex, rng)
            raw = _corrupt(ex, mode, rng)

        # Post-trained adapters are also shorter, which shows up in the latency
        # and token columns of the report.
        verbosity = 1.0 if adapter == "base" else 0.72
        completion = max(4, int(len(raw.split()) * 1.3 * verbosity))
        latency = (110.0 + 2.4 * completion) * (1.0 + rng.uniform(-0.12, 0.12))

        _ = time.perf_counter() - t0
        return Prediction(
            example_id=ex.id,
            task=ex.task,
            raw=raw,
            latency_ms=latency,
            prompt_tokens=max(8, len(ex.prompt.split()) * 2 + 340 * max(1, len(ex.frames))),
            completion_tokens=completion,
            adapter=adapter,
        )

    def chat(self, prompt: str, **_: Any) -> str:
        """Text-only completion. Drives the tool-calling agent in `serve/`.

        This simulates a *competent* tool-caller, not a random one: it reads the
        running transcript, picks a tool it has not already used, and stops once
        it has evidence. That is deliberate -- an agent loop whose guardrails are
        only ever exercised by a model emitting garbage tells you nothing about
        whether the loop works. The interesting cases (repeat suppression, budget
        exhaustion) need a model that behaves plausibly and *still* goes wrong.

        The one built-in pathology: on a question it cannot serve with the
        available tools it will keep searching rather than admit defeat, which is
        exactly the behaviour the step budget exists to contain.
        """
        low = prompt.lower()

        # Not the agent transcript -- fall back to an opaque completion.
        if "question:" not in low:
            rng = random.Random(_seed(str(self.seed), prompt[:256]))
            return "Mock response " + str(rng.randrange(10**6))

        question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
        q_low = question.lower()

        searched = "tool search_events returned" in low
        counted = "tool count_objects returned" in low
        described = "tool describe_scene returned" in low
        wants_count = any(w in q_low for w in ("how many", "count", "number of"))

        if wants_count and not counted:
            term = next(
                (w for w in ("forklift", "pallet", "worker", "cart", "cone") if w in q_low),
                "forklift",
            )
            return json.dumps({"tool": "count_objects", "arguments": {"term": term, "group_by_camera": True}})

        if not searched:
            terms = [w.strip("?,.") for w in question.split() if len(w) > 3][:6]
            args: dict[str, Any] = {"query": " ".join(terms), "limit": 5}
            for cam in CAMERA_HINTS:
                if cam in q_low:
                    args["camera"] = cam
            return json.dumps({"tool": "search_events", "arguments": args})

        # Occasionally take a second hop into the VLM before answering, which is
        # what makes multi-tool traces show up in the demo.
        if not described and "describe" in q_low:
            ids = re.findall(r'"id":\s*"([0-9a-f]{12})"', prompt)
            if ids:
                return json.dumps({"tool": "describe_scene", "arguments": {"id": ids[0]}})

        cams = sorted(set(re.findall(r'"camera":\s*"([a-z0-9-]+)"', prompt)))
        if counted:
            total = re.search(r'"total":\s*(\d+)', prompt)
            n = total.group(1) if total else "several"
            return json.dumps(
                {"answer": f"{n} matching records, spread across cameras: {', '.join(cams) or 'unknown'}."}
            )
        if cams:
            return json.dumps(
                {"answer": f"Evidence found on {len(cams)} camera(s): {', '.join(cams)}."}
            )
        return json.dumps({"answer": "No matching records were found in the indexed corpus."})

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Hashed bag-of-words. Lexically similar strings land near each other,
        which is enough to exercise the retrieval path without a network call."""
        dim = 256
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for tok in text.lower().split():
                h = int(hashlib.sha1(tok.encode()).hexdigest()[:8], 16)
                vec[h % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
