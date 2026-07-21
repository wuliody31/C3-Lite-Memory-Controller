from __future__ import annotations

import pytest

from evaluation.metrics import (
    aggregate,
    per_question_metrics,
)


def make_metrics(
    *,
    selected_memory_types: list[str],
    expected_memory_types: list[str],
    route_metric_applicable: bool,
) -> dict[str, float]:
    return per_question_metrics(
        selected_memory_types=selected_memory_types,
        expected_memory_types=expected_memory_types,
        route_metric_applicable=route_metric_applicable,
        used_ids=[],
        gold_ids=[],
        decision="abstain",
        should_abstain=True,
        expected_outdated_ids=[],
        retrieved_ids=[],
    )


def test_route_uses_explicit_per_question_labels():
    metrics = make_metrics(
        selected_memory_types=["episodic"],
        expected_memory_types=["episodic"],
        route_metric_applicable=True,
    )

    assert metrics["route_scored"] == 1.0
    assert metrics["route_exact"] == 1.0
    assert metrics["route_precision"] == 1.0
    assert metrics["route_recall"] == 1.0
    assert metrics["route_f1"] == 1.0


def test_multi_memory_route_is_scored_as_a_set():
    metrics = make_metrics(
        selected_memory_types=[
            "semantic",
            "episodic",
        ],
        expected_memory_types=[
            "episodic",
            "semantic",
            "procedural",
        ],
        route_metric_applicable=True,
    )

    assert metrics["route_exact"] == 0.0
    assert metrics["route_precision"] == 1.0
    assert metrics["route_recall"] == pytest.approx(
        2 / 3
    )
    assert metrics["route_f1"] == pytest.approx(
        0.8
    )


def test_unscored_route_is_omitted_not_zeroed():
    metrics = make_metrics(
        selected_memory_types=["procedural"],
        expected_memory_types=[],
        route_metric_applicable=False,
    )

    assert metrics["route_scored"] == 0.0
    assert "route_exact" not in metrics
    assert "route_precision" not in metrics
    assert "route_recall" not in metrics
    assert "route_f1" not in metrics


def test_missing_route_labels_are_not_scored():
    metrics = make_metrics(
        selected_memory_types=["semantic"],
        expected_memory_types=[],
        route_metric_applicable=True,
    )

    assert metrics["route_scored"] == 0.0
    assert "route_f1" not in metrics


def test_aggregate_route_denominator_excludes_unscored_rows():
    scored = make_metrics(
        selected_memory_types=["episodic"],
        expected_memory_types=["episodic"],
        route_metric_applicable=True,
    )
    unscored = make_metrics(
        selected_memory_types=["procedural"],
        expected_memory_types=[],
        route_metric_applicable=False,
    )

    summary = aggregate([scored, unscored])

    assert summary["route_scored"] == 0.5
    assert summary["route_f1"] == 1.0
    assert summary["route_exact"] == 1.0
