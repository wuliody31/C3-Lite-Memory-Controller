from __future__ import annotations

import csv

from evaluation.dataset_loader import load_evaluation_csv


def test_loads_per_question_route_labels(tmp_path):
    path = tmp_path / "evaluation.csv"

    fieldnames = [
        "question_id",
        "user_id",
        "question_type",
        "question",
        "supporting_memory_ids",
        "should_abstain",
        "expected_outdated_memory_ids",
        "conflict_type",
        "gold_answer",
        "expected_memory_types",
        "route_metric_applicable",
        "summary_expected_memory_types",
        "route_label_source",
        "route_label_note",
    ]

    rows = [
        {
            "question_id": "q1",
            "user_id": "user01",
            "question_type": "memory_selection",
            "question": "What changed?",
            "supporting_memory_ids": "e_1;s_1;p_1",
            "should_abstain": "False",
            "expected_outdated_memory_ids": "s_old",
            "conflict_type": "temporal",
            "gold_answer": "The scope changed.",
            "expected_memory_types": (
                "episodic;semantic;procedural"
            ),
            "route_metric_applicable": "True",
            "summary_expected_memory_types": (
                "episodic;semantic"
            ),
            "route_label_source": "gold_supporting_ids",
            "route_label_note": "Gold-derived route.",
        },
        {
            "question_id": "q2",
            "user_id": "user01",
            "question_type": "abstention",
            "question": "What is unsupported?",
            "supporting_memory_ids": "",
            "should_abstain": "True",
            "expected_outdated_memory_ids": "",
            "conflict_type": "none",
            "gold_answer": "",
            "expected_memory_types": "",
            "route_metric_applicable": "False",
            "summary_expected_memory_types": "",
            "route_label_source": (
                "not_applicable_no_gold_evidence"
            ),
            "route_label_note": "",
        },
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    questions = load_evaluation_csv(path)

    assert len(questions) == 2

    assert questions[0].expected_memory_types == [
        "episodic",
        "semantic",
        "procedural",
    ]
    assert questions[0].route_metric_applicable is True
    assert questions[0].summary_expected_memory_types == [
        "episodic",
        "semantic",
    ]
    assert (
        questions[0].route_label_source
        == "gold_supporting_ids"
    )

    assert questions[1].expected_memory_types == []
    assert questions[1].route_metric_applicable is False
    assert questions[1].should_abstain is True


def test_old_evaluation_csv_remains_compatible(tmp_path):
    path = tmp_path / "old_evaluation.csv"

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "user_id",
                "question_type",
                "question",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q_old",
                "user_id": "user01",
                "question_type": "memory_selection",
                "question": "What is the current focus?",
            }
        )

    question = load_evaluation_csv(path)[0]

    assert question.expected_memory_types == []
    assert question.route_metric_applicable is True
    assert question.summary_expected_memory_types == []
    assert question.route_label_source == ""
