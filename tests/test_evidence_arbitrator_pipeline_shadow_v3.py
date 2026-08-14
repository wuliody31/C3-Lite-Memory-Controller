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
        config[
            "query_analysis"
        ][
            "stopwords"
        ]
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


def run_query(
    query: str,
):
    pipeline = build_pipeline()

    try:
        return pipeline.answer(
            QueryState(
                query=query,
                user_id="user01",
            )
        )
    finally:
        pipeline.close()


def shadow(
    result,
) -> dict:
    return result.debug[
        "c3_v3_arbitration_shadow"
    ]


def test_current_query_active_arbitration_preserves_single_valid_evidence() -> None:
    result = run_query(
        "What is my current MSc project scope?"
    )

    trace = shadow(
        result
    )

    assert trace[
        "active_for_generation"
    ] is True

    assert trace[
        "active_selector"
    ] == "c3_v3_evidence_arbitrator"

    assert trace[
        "comparison_stage"
    ] == "post_resolution_pre_repair"

    assert trace[
        "legacy_selected_ids"
    ] == [
        "s_user01_003"
    ]

    assert trace[
        "c3_v3_selected_ids"
    ] == [
        "s_user01_003"
    ]

    assert trace[
        "same_set"
    ] is True

    assert trace[
        "hard_complete"
    ] is True

    # Shadow mode must not alter actual generation evidence.
    assert result.selected_ids == [
        "s_user01_003"
    ]


def test_historical_active_arbitration_removes_temporally_incompatible_current_filler() -> None:
    result = run_query(
        "What was my previous MSc project scope?"
    )

    trace = shadow(
        result
    )

    assert trace[
        "legacy_selected_ids"
    ] == [
        "s_user01_004",
        "s_user01_003",
    ]

    assert trace[
        "c3_v3_selected_ids"
    ] == [
        "s_user01_004"
    ]

    assert trace[
        "removed_by_c3_v3"
    ] == [
        "s_user01_003"
    ]

    assert (
        "s_user01_003"
        in trace[
            "rejected_incompatible"
        ]
    )

    assert trace[
        "hard_complete"
    ] is True

    # C3-v3 arbitration is now active for generation.
    assert trace[
        "active_for_generation"
    ] is True

    assert trace[
        "active_selector"
    ] == "c3_v3_evidence_arbitrator"

    assert result.selected_ids == [
        "s_user01_004",
    ]


def test_timeline_shadow_preserves_requirement_complete_evidence_set() -> None:
    result = run_query(
        "How did my project scope change over time?"
    )

    trace = shadow(
        result
    )

    legacy = set(
        trace[
            "legacy_selected_ids"
        ]
    )

    c3 = set(
        trace[
            "c3_v3_selected_ids"
        ]
    )

    expected = {
        "s_user01_003",
        "s_user01_004",
        "e_user01_007",
    }

    assert legacy == expected
    assert c3 == expected

    # Order is deliberately NOT asserted here.
    # Arbitration order and prompt presentation order
    # will be treated as separate concerns.
    assert trace[
        "same_set"
    ] is True

    assert trace[
        "hard_complete"
    ] is True

    assert trace[
        "soft_complete"
    ] is True

    assert trace[
        "removed_by_c3_v3"
    ] == []

    assert trace[
        "added_by_c3_v3"
    ] == []


def test_procedural_shadow_removes_zero_requirement_gain_filler() -> None:
    result = run_query(
        (
            "When discussing my dissertation "
            "with my supervisor, how should "
            "the answer be written?"
        )
    )

    trace = shadow(
        result
    )

    assert trace[
        "legacy_selected_ids"
    ] == [
        "p_user01_001",
        "p_user01_003",
    ]

    assert trace[
        "c3_v3_selected_ids"
    ] == [
        "p_user01_001"
    ]

    assert trace[
        "removed_by_c3_v3"
    ] == [
        "p_user01_003"
    ]

    assert trace[
        "hard_complete"
    ] is True

    assert trace[
        "soft_complete"
    ] is True


def test_explanation_shadow_preserves_complete_multi_evidence_set() -> None:
    result = run_query(
        "Why did my MSc project scope change?"
    )

    trace = shadow(
        result
    )

    legacy = set(
        trace[
            "legacy_selected_ids"
        ]
    )

    c3 = set(
        trace[
            "c3_v3_selected_ids"
        ]
    )

    expected = {
        "s_user01_003",
        "s_user01_004",
        "e_user01_007",
    }

    assert legacy == expected
    assert c3 == expected

    assert trace[
        "same_set"
    ] is True

    assert trace[
        "hard_complete"
    ] is True

    assert trace[
        "soft_complete"
    ] is True

    # Transition evidence must survive the soft phase.
    assert (
        "e_user01_007"
        in trace[
            "c3_v3_selected_ids"
        ]
    )