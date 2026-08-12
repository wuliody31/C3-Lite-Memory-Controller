from __future__ import annotations

from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import (
    QueryMode,
    QueryState,
)


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


def test_c3_v3_trace_is_present() -> None:
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

    assert (
        result.debug[
            "c3_v3_shadow_only"
        ]
        is False
    )

    assert (
        "c3_v3_requirement_spec"
        in result.debug
    )

    assert (
        "c3_v3_sufficiency"
        in result.debug
    )

    assert (
        "c3_v3_repair_plan"
        in result.debug
    )


def test_shadow_compilation_matches_legacy() -> None:
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

    compatibility = result.debug[
        "c3_v3_legacy_compatibility"
    ]

    assert compatibility[
        "query_mode_match"
    ]

    assert compatibility[
        "route_match"
    ]

    assert compatibility[
        "information_needs_match"
    ]


def test_satisfied_query_does_not_plan_repair() -> None:
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

    sufficiency = result.debug[
        "c3_v3_sufficiency"
    ]

    repair = result.debug[
        "c3_v3_repair_plan"
    ]

    assert sufficiency[
        "sufficient"
    ]

    assert not repair[
        "needed"
    ]

def test_previous_scope_is_treated_as_historical() -> None:
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

    assert (
        result.query_mode
        == QueryMode.HISTORICAL
    )

    requirement_spec = result.debug[
        "c3_v3_requirement_spec"
    ]

    roles = {
        item["role"]
        for item
        in requirement_spec[
            "requirements"
        ]
    }

    assert "historical_state" in roles