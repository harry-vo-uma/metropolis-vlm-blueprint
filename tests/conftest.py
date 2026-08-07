"""Shared fixtures.

The mock backend is forced for every test. A suite that silently starts making
paid network calls because a developer happened to have a key exported is a bad
surprise, and a non-deterministic test suite is worse than no test suite.
"""

from __future__ import annotations

import os

os.environ["MVB_FORCE_MOCK"] = "1"
os.environ.setdefault("MVB_TRACE_ENABLED", "0")

import pytest  # noqa: E402

from mvb.config import reset_settings  # noqa: E402
from mvb.data.synth import generate_pool  # noqa: E402
from mvb.nim.client import reset_backend  # noqa: E402
from mvb.nim.mock import MockVLMBackend  # noqa: E402
from mvb.schemas import Example, Frame, Provenance, Split, TaskKind  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    reset_settings()
    reset_backend()
    yield
    reset_settings()
    reset_backend()


@pytest.fixture
def backend():
    return MockVLMBackend()


@pytest.fixture
def pool():
    return generate_pool(n=300, seed=7)


def make_example(
    task: TaskKind = TaskKind.SCENE_QA,
    prompt: str = "What is happening?",
    target: str = "A forklift is idle.",
    **kwargs,
) -> Example:
    return Example(
        id=kwargs.pop("id", Example.make_id(task, prompt, target)),
        task=task,
        split=kwargs.pop("split", Split.TEST),
        provenance=kwargs.pop("provenance", Provenance.HUMAN_LABELLED),
        frames=kwargs.pop("frames", [Frame(uri="frames/dock-a/00001_0.jpg", camera="dock-a")]),
        prompt=prompt,
        target=target,
        **kwargs,
    )


@pytest.fixture
def example_factory():
    return make_example
