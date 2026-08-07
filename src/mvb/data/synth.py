"""Generator for the shipped evaluation suite.

Produces a reproducible pool of warehouse-domain multimodal examples across the
five tasks, with tag slices (`night`, `occluded`, `small_object`, `crowded`,
`motion_blur`) and a difficulty distribution that is not uniform -- real
collected sets are long-tailed, and a uniform one makes every slice look
equally tractable.

To be explicit about what this is: the *frames are references, not pixels*. The
suite ships as text so it can live in git, be diffed, and be reviewed. Pointing
`MVB_FRAME_ROOT` at a real image corpus swaps in real frames without touching
the generator, and `docs/evaluation.md` states plainly which conclusions survive
that swap and which do not.
"""

from __future__ import annotations

import json
import random

from ..schemas import Example, Frame, Provenance, Split, TaskKind

CAMERAS = ["dock-a", "dock-b", "aisle-03", "aisle-07", "pack-01", "pack-03", "yard-n"]

_OBJECTS = [
    ("forklift", "forklifts"),
    ("pallet", "pallets"),
    ("worker", "workers"),
    ("cart", "carts"),
    ("stack of boxes", "stacks of boxes"),
    ("safety cone", "safety cones"),
]

_ZONES = [
    "the loading dock",
    "the keep-clear aisle",
    "packing station 3",
    "the staging area",
    "the inbound lane",
    "the pallet wrap bay",
    "the charging bank",
]

_COLOURS = ["yellow", "orange", "blue", "grey", "red", "green", "white"]
_STATES = ["idle", "moving", "loading", "unloading", "parked", "reversing", "queued"]

_RELATIONS = ["left of", "right of", "above", "below", "in front of", "behind", "inside", "outside"]

_TAG_POOL = ["night", "occluded", "small_object", "crowded", "motion_blur"]


def _frame(rng: random.Random, idx: int, n: int = 1) -> list[Frame]:
    cam = rng.choice(CAMERAS)
    t0 = round(rng.uniform(0, 43200), 1)
    return [
        Frame(
            uri=f"frames/{cam}/{idx:05d}_{k}.jpg",
            camera=cam,
            timestamp_s=t0 + k * 1.5,
        )
        for k in range(n)
    ]


def _tags(rng: random.Random) -> list[str]:
    """Long-tailed slice assignment. Most frames are clean; the hard slices
    overlap, and the overlap is where the model actually lives."""
    roll = rng.random()
    if roll < 0.46:
        return []
    if roll < 0.82:
        return [rng.choice(_TAG_POOL)]
    return sorted(rng.sample(_TAG_POOL, 2))


def _difficulty(rng: random.Random, tags: list[str]) -> float:
    base = rng.betavariate(2.2, 3.0)  # skewed easy, with a real tail
    return round(min(0.98, base + 0.11 * len(tags)), 3)


def _scene_qa(rng: random.Random, idx: int) -> tuple[str, str, str | None]:
    sing, plur = rng.choice(_OBJECTS)
    zone = rng.choice(_ZONES)
    colour = rng.choice(_COLOURS)
    state = rng.choice(_STATES)
    n = rng.randint(1, 5)
    style = rng.randrange(4)
    if style == 0:
        return (
            f"What is the {colour} {sing} doing near {zone}?",
            f"The {colour} {sing} is {state} near {zone}.",
            None,
        )
    if style == 1:
        return (
            f"Is the {colour} {sing} obstructing {zone}?",
            f"The {colour} {sing} is {state} and does not obstruct {zone}; the marked boundary stays clear.",
            None,
        )
    if style == 2:
        return (
            f"How many {plur} are visible near {zone}?",
            f"{n} {plur if n > 1 else sing} {'are' if n > 1 else 'is'} visible near {zone}, {state}.",
            None,
        )
    return (
        f"Describe the activity at {zone} involving the {colour} {sing}.",
        f"A {colour} {sing} is {state} while {n} {plur} wait alongside {zone}.",
        None,
    )


def _attribute(rng: random.Random, idx: int) -> tuple[str, str, str | None]:
    sing, _ = rng.choice(_OBJECTS)
    payload = {
        "object": sing,
        "colour": rng.choice(_COLOURS),
        "state": rng.choice(_STATES),
        "count": rng.randint(1, 6),
    }
    if rng.random() < 0.4:
        payload["zone"] = rng.choice(_ZONES)
    prompt = (
        f"Extract the attributes of the {sing} in this frame as JSON with keys "
        f"{sorted(payload)}."
    )
    return prompt, json.dumps(payload, sort_keys=True), None


def _spatial(rng: random.Random, idx: int) -> tuple[str, str, str | None]:
    (a, _), (b, _) = rng.sample(_OBJECTS, 2)
    rel = rng.choice(_RELATIONS)
    colour = rng.choice(_COLOURS)
    zone = rng.choice(_ZONES)
    return (
        f"Where is the {colour} {a} relative to the {b} at {zone}?",
        f"The {colour} {a} is {rel} the {b}.",
        None,
    )


def _anomaly(rng: random.Random, idx: int) -> tuple[str, str, str | None]:
    sing, _ = rng.choice(_OBJECTS)
    zone = rng.choice(_ZONES)
    colour = rng.choice(_COLOURS)
    dwell = rng.randrange(5, 240, 5)
    # The verdict is a function of the observation, not a coin flip. A judgement
    # set whose label is independent of the prompt trains the model to guess the
    # prior, and no amount of eval slicing will reveal that it did.
    anomalous = dwell >= 90
    verdict = "Yes" if anomalous else "No"
    rationale = (
        f"the {colour} {sing} has held position in {zone} for {dwell} seconds with no operator present"
        if anomalous
        else f"the {colour} {sing} has only been in {zone} for {dwell} seconds, within the {zone} dwell allowance"
    )
    return (
        f"A {colour} {sing} has been stationary in {zone} for {dwell} seconds. "
        "Is this anomalous? Give a verdict and justify it.",
        verdict,
        rationale,
    )


_EVENT_LABELS = [
    "forklift_enters",
    "pallet_lowered",
    "worker_approaches",
    "gate_opens",
    "cart_departs",
    "aisle_clears",
    "scanner_beeps",
    "wrap_cycle_starts",
    "dock_door_closes",
]


def _temporal(rng: random.Random, idx: int) -> tuple[str, str, str | None]:
    events = rng.sample(_EVENT_LABELS, k=rng.randint(3, 5))
    zone = rng.choice(_ZONES)
    return (
        f"These events occurred at {zone}. Order them chronologically and return a "
        "JSON array of the labels: " + ", ".join(sorted(events)),
        json.dumps(events),
        None,
    )


_BUILDERS = {
    TaskKind.SCENE_QA: _scene_qa,
    TaskKind.ATTRIBUTE_EXTRACTION: _attribute,
    TaskKind.SPATIAL_RELATION: _spatial,
    TaskKind.ANOMALY_JUDGEMENT: _anomaly,
    TaskKind.TEMPORAL_ORDERING: _temporal,
}

#: Raw pool composition, before curation. Deliberately unbalanced -- SCENE_QA
#: dominates because open-ended captions are the cheapest thing to label, which
#: is exactly the skew `curate.balance_tasks` exists to correct.
_TASK_WEIGHTS = {
    TaskKind.SCENE_QA: 0.38,
    TaskKind.ATTRIBUTE_EXTRACTION: 0.20,
    TaskKind.SPATIAL_RELATION: 0.17,
    TaskKind.ANOMALY_JUDGEMENT: 0.15,
    TaskKind.TEMPORAL_ORDERING: 0.10,
}

_PROVENANCE_WEIGHTS = {
    Provenance.HUMAN_LABELLED: 0.46,
    Provenance.SYNTHETIC_AUGMENTED: 0.28,
    Provenance.MODEL_DISTILLED: 0.16,
    Provenance.RULE_DERIVED: 0.10,
}


def generate_pool(n: int = 2100, seed: int = 1337) -> list[Example]:
    """Generate the raw pool. Curation trims this to the shipped suite.

    The default of 2100 is chosen so that after deduplication and task balancing
    the result lands at the 1,500-example suite size.
    """
    rng = random.Random(seed)
    tasks = list(_TASK_WEIGHTS)
    task_w = [_TASK_WEIGHTS[t] for t in tasks]
    provs = list(_PROVENANCE_WEIGHTS)
    prov_w = [_PROVENANCE_WEIGHTS[p] for p in provs]

    out: list[Example] = []
    for i in range(n):
        task = rng.choices(tasks, weights=task_w, k=1)[0]
        prompt, target, rationale = _BUILDERS[task](rng, i)
        tags = _tags(rng)
        n_frames = 3 if task is TaskKind.TEMPORAL_ORDERING else 1
        ex = Example(
            id=Example.make_id(task, prompt, target),
            task=task,
            split=Split.TRAIN,
            provenance=rng.choices(provs, weights=prov_w, k=1)[0],
            frames=_frame(rng, i, n_frames),
            prompt=prompt,
            target=target,
            rationale=rationale,
            tags=tags,
            difficulty=_difficulty(rng, tags),
        )
        out.append(ex)
    return out
