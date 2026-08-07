from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "lifecycle_benchmark_v01"
    / "cases.jsonl"
)


def load_cases():
    return [
        json.loads(line)
        for line in DATA.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def test_benchmark_has_100_unique_cases():
    cases = load_cases()

    assert len(cases) == 100

    ids = [
        case["case_id"]
        for case in cases
    ]

    assert len(ids) == len(set(ids))


def test_benchmark_is_balanced():
    cases = load_cases()

    patterns = Counter(
        case["pattern"]
        for case in cases
    )

    state_keys = Counter(
        case["state_key"]
        for case in cases
    )

    assert patterns == {
        "replacement": 20,
        "duplicate": 20,
        "multi_step": 20,
        "out_of_order": 20,
        "reversion": 20,
    }

    assert len(state_keys) == 5

    assert set(
        state_keys.values()
    ) == {20}


def test_event_timestamps_are_valid():
    for case in load_cases():
        for item in case["events"]:
            value = datetime.fromisoformat(
                item["observed_at"]
            )

            assert value.tzinfo is not None


def test_operation_contracts_match_patterns():
    expected = {
        "replacement": [
            "add_initial",
            "supersede_state",
        ],
        "duplicate": [
            "add_initial",
            "noop_duplicate",
        ],
        "multi_step": [
            "add_initial",
            "supersede_state",
            "supersede_state",
        ],
        "out_of_order": [
            "add_initial",
            "supersede_state",
            "reject_out_of_order",
        ],
        "reversion": [
            "add_initial",
            "supersede_state",
            "supersede_state",
        ],
    }

    for case in load_cases():
        assert (
            case["expected_operations"]
            == expected[case["pattern"]]
        )


def test_current_gold_is_never_stale():
    for case in load_cases():
        assert (
            case["current_gold"]
            not in case["stale_values"]
        )


def test_duplicate_has_no_stale_history():
    for case in load_cases():
        if case["pattern"] != "duplicate":
            continue

        assert case["historical_gold"] == []
        assert case["stale_values"] == []


def test_out_of_order_uses_older_final_observation():
    for case in load_cases():
        if case["pattern"] != "out_of_order":
            continue

        second = datetime.fromisoformat(
            case["events"][1]["observed_at"]
        )

        third = datetime.fromisoformat(
            case["events"][2]["observed_at"]
        )

        assert third < second


def test_reversion_returns_to_initial_value():
    for case in load_cases():
        if case["pattern"] != "reversion":
            continue

        assert (
            case["events"][0]["value"]
            == case["events"][-1]["value"]
        )

        assert (
            case["current_gold"]
            == case["events"][0]["value"]
        )



def test_multi_step_distinguishes_previous_from_full_history():
    for case in load_cases():
        if case["pattern"] != "multi_step":
            continue

        values = [
            item["value"]
            for item in case["events"]
        ]

        assert case["historical_gold"] == [
            values[1]
        ]

        assert (
            case["history_values_expected"]
            == values[:2]
        )


def test_reversion_preserves_earlier_same_value_history():
    for case in load_cases():
        if case["pattern"] != "reversion":
            continue

        values = [
            item["value"]
            for item in case["events"]
        ]

        assert values[0] == values[2]

        assert case["historical_gold"] == [
            values[1]
        ]

        assert (
            case["history_values_expected"]
            == values[:2]
        )
