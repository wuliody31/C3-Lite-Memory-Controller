from __future__ import annotations

from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import QueryState


ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> C3Pipeline:
    config = load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )

    stopwords = set(
        config["query_analysis"]["stopwords"]
    )

    memory_store = (
        InMemoryMemoryStore.from_json_file(
            str(
                ROOT
                / "examples"
                / "demo_memories.json"
            ),
            stopwords,
        )
    )

    procedure_store = (
        ProceduralJsonStore.from_file(
            str(
                ROOT
                / "examples"
                / "demo_procedures.json"
            ),
            stopwords,
        )
    )

    return C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=MockBackbone(),
        prompt_template=(
            ROOT
            / "prompts"
            / "answer_prompt.txt"
        ),
    )


def test_current_query_temporal_validity_prefers_current() -> None:
    pipeline = build_pipeline()

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What is my current "
                    "MSc project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    trace = result.debug[
        "c3_v3_temporal_validity"
    ]

    assert "s_user01_003" in trace

    assert (
        trace[
            "s_user01_003"
        ][
            "new_query_relative_validity"
        ]
        >= 0.90
    )


def test_historical_query_prefers_historical_version() -> None:
    pipeline = build_pipeline()

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What was my previous "
                    "MSc project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    trace = result.debug[
        "c3_v3_temporal_validity"
    ]

    assert "s_user01_004" in trace
    assert "s_user01_003" in trace

    historical = trace[
        "s_user01_004"
    ][
        "new_query_relative_validity"
    ]

    current = trace[
        "s_user01_003"
    ][
        "new_query_relative_validity"
    ]

    assert historical > current

    assert (
        trace[
            "s_user01_004"
        ][
            "temporal_role"
        ]
        == "historical_state"
    )


def test_timeline_assigns_different_temporal_roles() -> None:
    pipeline = build_pipeline()

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "How did my project scope "
                    "change over time?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    trace = result.debug[
        "c3_v3_temporal_validity"
    ]

    assert "s_user01_003" in trace
    assert "s_user01_004" in trace

    current = trace[
        "s_user01_003"
    ]

    historical = trace[
        "s_user01_004"
    ]

    assert current["compatible"]
    assert historical["compatible"]

    assert (
        current["temporal_role"]
        == "current_endpoint"
    )

    assert (
        historical["temporal_role"]
        == "historical_state"
    )

    assert (
        current[
            "new_query_relative_validity"
        ]
        >= 0.90
    )

    assert (
        historical[
            "new_query_relative_validity"
        ]
        >= 0.90
    )


def test_old_and_new_validity_are_both_traced() -> None:
    pipeline = build_pipeline()

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What was my previous "
                    "MSc project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    trace = result.debug[
        "c3_v3_temporal_validity"
    ]

    historical = trace[
        "s_user01_004"
    ]

    assert (
        "old_validity_score"
        in historical
    )

    assert (
        "new_query_relative_validity"
        in historical
    )

    # This is the key diagnostic difference:
    # old RC8 validity and C3-v3 query-relative
    # validity are explicitly observable together.