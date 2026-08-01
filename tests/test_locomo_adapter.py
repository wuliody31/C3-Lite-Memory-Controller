from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_locomo_dataset import prepare


def write_source(
    path: Path,
    data: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def minimal_sample(
    *,
    sample_id: str = "conv-test",
) -> dict:
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": (
                "1:56 pm on 8 May, 2023"
            ),
            "session_1": [
                {
                    "speaker": "Alice",
                    "dia_id": "D1:1",
                    "text": "I enjoy drinking tea.",
                },
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "I prefer coffee.",
                },
            ],
        },
        "observation": {
            "session_1_observation": {
                "Alice": [
                    [
                        "Alice enjoys drinking tea.",
                        "D1:1",
                    ]
                ]
            }
        },
        "qa": [
            {
                "question": (
                    "What does Alice enjoy drinking?"
                ),
                "answer": "Tea",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": (
                    "What is Alice's favourite city?"
                ),
                "answer": None,
                "evidence": ["D1:2"],
                "category": 2,
            },
        ],
    }


def test_prepare_maps_dialogue_and_observation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "locomo.json"
    output = tmp_path / "output"

    write_source(
        source,
        [minimal_sample()],
    )

    manifest = prepare(
        source=source,
        output_dir=output,
        include_observations=True,
        smoke_size=2,
        pilot_size=2,
        seed=20260801,
        strict=True,
    )

    memories = json.loads(
        (
            output / "memories_locomo.json"
        ).read_text(encoding="utf-8")
    )["memories"]

    memory_map = {
        item["memory_id"]: item
        for item in memories
    }

    episodic_id = (
        "e_locomo_conv_test_d1_1"
    )
    semantic_id = (
        "s_locomo_conv_test_"
        "obs_s01_alice_001"
    )

    assert episodic_id in memory_map
    assert semantic_id in memory_map

    assert (
        memory_map[semantic_id]["source_ids"]
        == [episodic_id]
    )

    questions = read_csv(
        output
        / "eval_questions_locomo_all.csv"
    )

    assert len(questions) == 2
    assert (
        questions[0]["supporting_memory_ids"]
        == episodic_id
    )
    assert (
        questions[0]["route_metric_applicable"]
        == "False"
    )
    assert (
        questions[1]["should_abstain"]
        == "True"
    )

    assert (
        manifest["counts"][
            "episodic_memories"
        ]
        == 2
    )
    assert (
        manifest["counts"][
            "semantic_memories"
        ]
        == 1
    )


def test_namespaces_repeated_dialogue_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "locomo.json"
    output = tmp_path / "output"

    first = minimal_sample(
        sample_id="conv-one"
    )
    second = minimal_sample(
        sample_id="conv-two"
    )

    write_source(
        source,
        [first, second],
    )

    prepare(
        source=source,
        output_dir=output,
        include_observations=False,
        smoke_size=0,
        pilot_size=0,
        seed=20260801,
        strict=True,
    )

    memories = json.loads(
        (
            output / "memories_locomo.json"
        ).read_text(encoding="utf-8")
    )["memories"]

    memory_ids = [
        item["memory_id"]
        for item in memories
    ]

    assert len(memory_ids) == 4
    assert len(set(memory_ids)) == 4

    assert (
        "e_locomo_conv_one_d1_1"
        in memory_ids
    )
    assert (
        "e_locomo_conv_two_d1_1"
        in memory_ids
    )


def test_strict_mode_rejects_missing_gold_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "locomo.json"
    output = tmp_path / "output"

    sample = minimal_sample()
    sample["qa"][0]["evidence"] = [
        "D99:99"
    ]

    write_source(
        source,
        [sample],
    )

    with pytest.raises(
        RuntimeError,
        match="Strict validation failed",
    ):
        prepare(
            source=source,
            output_dir=output,
            include_observations=False,
            smoke_size=0,
            pilot_size=0,
            seed=20260801,
            strict=True,
        )
