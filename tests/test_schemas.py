import json

import pytest
from conftest import make_example

from mvb.schemas import Example, TaskKind


def test_empty_target_is_rejected():
    """An empty reference silently scores zero forever; fail loudly instead."""
    with pytest.raises(ValueError):
        make_example(target="   ")


def test_ids_are_stable_across_processes():
    a = Example.make_id(TaskKind.SCENE_QA, "p", "t")
    b = Example.make_id(TaskKind.SCENE_QA, "p", "t")
    assert a == b and len(a) == 12


def test_structured_tasks_parse_their_target():
    ex = make_example(task=TaskKind.ATTRIBUTE_EXTRACTION, target=json.dumps({"count": 3}))
    assert ex.is_structured()
    assert ex.parsed_target() == {"count": 3}


def test_freeform_target_is_returned_verbatim():
    ex = make_example(target="A forklift is idle.")
    assert not ex.is_structured()
    assert ex.parsed_target() == "A forklift is idle."
