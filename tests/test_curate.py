"""Curation is a series of decisions. Each one should be defensible in a test."""

from __future__ import annotations

from mvb.data.curate import curate, deduplicate, split_examples, trim_to_size
from mvb.data.synth import generate_pool
from mvb.schemas import Split


def test_duplicates_are_removed_and_counted() -> None:
    pool = generate_pool(n=400, seed=7)
    kept, dropped = deduplicate(pool + pool)
    assert dropped >= len(pool)
    assert len({e.id for e in kept}) == len(kept)


def test_trim_lands_on_the_exact_target() -> None:
    pool = generate_pool(n=600, seed=11)
    kept, dropped = trim_to_size(pool, 500)
    assert len(kept) == 500
    assert dropped == len(pool) - 500


def test_trim_is_a_no_op_when_already_small_enough() -> None:
    pool = generate_pool(n=100, seed=11)
    kept, dropped = trim_to_size(pool, 500)
    assert len(kept) == len(pool)
    assert dropped == 0


def test_splits_are_stable_when_the_pool_grows() -> None:
    """Hash assignment, not shuffling.

    Shuffle-based splitting re-deals the whole deck whenever data is added,
    which quietly moves last month's test examples into train. Here an example
    must land in the same split regardless of what surrounds it.
    """
    small = generate_pool(n=300, seed=3)
    large = small + generate_pool(n=300, seed=4)

    before = {e.id: e.split for e in split_examples(small)}
    after = {e.id: e.split for e in split_examples(large)}

    shared = set(before) & set(after)
    assert shared
    assert all(before[i] is after[i] for i in shared)


def test_curate_hits_the_requested_size_and_reports_every_stage() -> None:
    pool = generate_pool(n=900, seed=5)
    rows, report = curate(pool, seed=5, target_size=600)
    assert len(rows) == 600
    assert report.input_n == len(pool)
    assert report.output_n == 600
    # The report must account for everything it removed, or it is decoration.
    removed = report.dropped_duplicates + report.dropped_provenance + report.dropped_balance + report.dropped_trim
    assert report.input_n - removed == report.output_n


def test_every_split_is_represented() -> None:
    pool = generate_pool(n=900, seed=5)
    rows, _ = curate(pool, seed=5, target_size=600)
    present = {r.split for r in rows}
    assert present == set(Split)
