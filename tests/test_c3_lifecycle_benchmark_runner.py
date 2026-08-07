from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPT = (
    ROOT
    / "scripts"
    / "phase2b"
    / "run_c3_lifecycle_benchmark_v01.py"
)

CASES = (
    ROOT
    / "data"
    / "lifecycle_benchmark_v01"
    / "cases.jsonl"
)


def run_benchmark(tmp_path: Path):
    output = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cases",
            str(CASES),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(
        (
            output
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    results = [
        json.loads(line)
        for line in (
            output
            / "results.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    return summary, results


def test_c3_lifecycle_contract_metrics(tmp_path):
    summary, results = run_benchmark(
        tmp_path
    )

    overall = summary["overall"]

    assert len(results) == 100
    assert overall["case_count"] == 100

    assert (
        overall[
            "operation_exact_match_rate"
        ]
        == 1.0
    )

    assert (
        overall[
            "operation_step_accuracy"
        ]
        == 1.0
    )

    assert (
        overall[
            "current_state_accuracy"
        ]
        == 1.0
    )

    assert (
        overall[
            "single_current_invariant_rate"
        ]
        == 1.0
    )

    assert (
        overall[
            "stale_memory_exposure_rate"
        ]
        == 0.0
    )

    assert (
        overall[
            "previous_state_recall"
        ]
        == 1.0
    )

    assert (
        overall[
            "history_retention_recall"
        ]
        == 1.0
    )

    assert (
        overall[
            "contradiction_rate"
        ]
        == 0.0
    )


def test_c3_lifecycle_operation_counts(tmp_path):
    summary, _ = run_benchmark(
        tmp_path
    )

    assert (
        summary["overall"][
            "operation_counts"
        ]
        == {
            "add_initial": 100,
            "noop_duplicate": 20,
            "reject_out_of_order": 20,
            "supersede_state": 120,
        }
    )


def test_c3_lifecycle_memory_growth(tmp_path):
    summary, _ = run_benchmark(
        tmp_path
    )

    overall = summary["overall"]

    assert (
        overall[
            "mean_semantic_records"
        ]
        == 2.2
    )

    assert (
        overall[
            "mean_current_semantic_records"
        ]
        == 1.0
    )

    assert (
        overall[
            "mean_superseded_semantic_records"
        ]
        == 1.2
    )

    assert (
        overall[
            "mean_episodic_transition_records"
        ]
        == 1.2
    )

    assert (
        overall[
            "mean_total_records"
        ]
        == 3.4
    )


def test_every_pattern_has_twenty_cases(tmp_path):
    summary, _ = run_benchmark(
        tmp_path
    )

    assert set(
        summary["by_pattern"]
    ) == {
        "replacement",
        "duplicate",
        "multi_step",
        "out_of_order",
        "reversion",
    }

    for row in (
        summary["by_pattern"].values()
    ):
        assert row["case_count"] == 20


def test_out_of_order_is_rejected(tmp_path):
    _, results = run_benchmark(
        tmp_path
    )

    rows = [
        row
        for row in results
        if row["pattern"]
        == "out_of_order"
    ]

    assert len(rows) == 20

    for row in rows:
        assert (
            row["actual_operations"][-1]
            == "reject_out_of_order"
        )

        assert (
            row["current_state_correct"]
            is True
        )


def test_reversion_preserves_history(tmp_path):
    _, results = run_benchmark(
        tmp_path
    )

    rows = [
        row
        for row in results
        if row["pattern"]
        == "reversion"
    ]

    assert len(rows) == 20

    for row in rows:
        assert (
            row[
                "history_retention_recall"
            ]
            == 1.0
        )

        assert (
            row[
                "previous_state_recall"
            ]
            == 1.0
        )

        assert (
            row["stale_exposure"]
            is False
        )
