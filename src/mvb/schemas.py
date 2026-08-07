"""Contracts shared by the data, training, evaluation, and serving stages.

Everything that crosses a stage boundary is a pydantic model. The point is not
validation for its own sake -- it is that a curation script, a training run, and
an eval harness written months apart can only interoperate if they agree on what
an "example" is. Most of the failure analysis in `docs/post-training.md` was only
possible because every record carries its provenance and its task tag.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskKind(str, Enum):
    """The five domain tasks the blueprint post-trains and evaluates against.

    These are deliberately narrow. A VLM that is 4% better at generic captioning
    is not a product; a VLM that reliably answers "is this aisle blocked" is.
    """

    SCENE_QA = "scene_qa"
    """Free-form question about a single frame."""

    ATTRIBUTE_EXTRACTION = "attribute_extraction"
    """Structured field extraction (colour, count, state) returned as JSON."""

    SPATIAL_RELATION = "spatial_relation"
    """Relative position / containment / occlusion between two referents."""

    ANOMALY_JUDGEMENT = "anomaly_judgement"
    """Binary judgement with a required justification."""

    TEMPORAL_ORDERING = "temporal_ordering"
    """Ordering of events across a short frame sequence."""


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Provenance(str, Enum):
    """Where an example came from.

    Kept on every record because the single most useful failure-analysis cut was
    "accuracy by provenance" -- synthetic-augmented examples were inflating the
    val score while doing nothing for the held-out human-labelled test set.
    """

    HUMAN_LABELLED = "human_labelled"
    SYNTHETIC_AUGMENTED = "synthetic_augmented"
    MODEL_DISTILLED = "model_distilled"
    RULE_DERIVED = "rule_derived"


class Frame(BaseModel):
    """A reference to an image, not the image itself.

    Datasets are shipped as JSONL; embedding base64 frames would make them
    unreadable and undiffable. The loader resolves `uri` at read time.
    """

    uri: str
    camera: str = "unknown"
    timestamp_s: float = 0.0
    width: int = 1920
    height: int = 1080


class Example(BaseModel):
    """One supervised multimodal example."""

    id: str
    task: TaskKind
    split: Split
    provenance: Provenance
    frames: list[Frame] = Field(default_factory=list)
    prompt: str
    target: str
    """Reference answer. For structured tasks this is a JSON string."""

    rationale: str | None = None
    """Optional chain of reasoning. Used for SFT on the judgement tasks only."""

    tags: list[str] = Field(default_factory=list)
    """Free-form slice labels: `night`, `occluded`, `small_object`, `crowded`."""

    difficulty: float = 0.5
    """0 = trivial, 1 = adversarial. Drives the curriculum in `curate.py`."""

    @field_validator("target")
    @classmethod
    def _target_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target must be non-empty; an empty reference silently scores 0")
        return v

    @staticmethod
    def make_id(task: TaskKind, prompt: str, target: str) -> str:
        raw = f"{task.value}|{prompt}|{target}".encode()
        return hashlib.sha1(raw).hexdigest()[:12]

    def is_structured(self) -> bool:
        return self.task in (TaskKind.ATTRIBUTE_EXTRACTION, TaskKind.TEMPORAL_ORDERING)

    def parsed_target(self) -> Any:
        if not self.is_structured():
            return self.target
        try:
            return json.loads(self.target)
        except json.JSONDecodeError:
            return self.target


class Prediction(BaseModel):
    example_id: str
    task: TaskKind
    raw: str
    """Exactly what the model emitted, before any parsing. Kept verbatim so that
    format failures can be distinguished from content failures."""

    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    adapter: str = "base"


class FailureMode(str, Enum):
    """Taxonomy from the failure analysis. Each mode has a distinct fix, which is
    the entire reason for separating them -- "wrong answer" is not actionable."""

    CORRECT = "correct"
    FORMAT_VIOLATION = "format_violation"
    """Right content, unparseable shape. Fixed by SFT on format, not by more data."""

    HALLUCINATED_OBJECT = "hallucinated_object"
    """Referred to something not in frame. Worst failure for an alerting product."""

    MISCOUNT = "miscount"
    SPATIAL_INVERSION = "spatial_inversion"
    """Left/right, above/below, in front/behind flipped."""

    OVERCAUTIOUS_REFUSAL = "overcautious_refusal"
    """Declined to answer a well-posed question. Common regression after safety SFT."""

    TEMPORAL_CONFUSION = "temporal_confusion"
    OTHER = "other"


class Grade(BaseModel):
    example_id: str
    task: TaskKind
    correct: bool
    score: float = 0.0
    """Task-appropriate partial credit in [0, 1]. `correct` is `score >= threshold`."""

    failure_mode: FailureMode = FailureMode.CORRECT
    detail: str = ""
    tags: list[str] = Field(default_factory=list)


class SliceResult(BaseModel):
    name: str
    n: int
    accuracy: float
    mean_score: float


class EvalReport(BaseModel):
    adapter: str
    n: int
    accuracy: float
    mean_score: float
    by_task: dict[str, SliceResult] = Field(default_factory=dict)
    by_tag: dict[str, SliceResult] = Field(default_factory=dict)
    by_provenance: dict[str, SliceResult] = Field(default_factory=dict)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def summary_line(self) -> str:
        return (
            f"{self.adapter:<12} n={self.n:<5} acc={self.accuracy:.3f} "
            f"score={self.mean_score:.3f} p95={self.p95_latency_ms:.0f}ms"
        )


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    ok: bool
    content: Any = None
    error: str | None = None
    latency_ms: float = 0.0
