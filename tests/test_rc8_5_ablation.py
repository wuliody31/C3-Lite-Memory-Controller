from __future__ import annotations

from pathlib import Path

import pytest

from src.ablation import (
    AblationSettings,
    force_all_memory_route,
    select_ranked_top_k,
)
from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
    QueryState,
)


ROOT = Path(__file__).resolve().parents[1]


CONFIG_CASES = {
    "rc85_no_route_planner.yaml": (
        "no_route_planner",
        "disable_route_planner",
    ),
    "rc85_no_conflict_handling.yaml": (
        "no_conflict_handling",
        "disable_conflict_handling",
    ),
    "rc85_no_coverage_confidence_gate.yaml": (
        "no_coverage_confidence_gate",
        "disable_coverage_confidence_gate",
    ),
    "rc85_no_evidence_selector.yaml": (
        "no_evidence_selector",
        "disable_evidence_selector",
    ),
}


def load_ablation_config(
    name: str,
) -> dict:
    return load_config(
        ROOT
        / "configs"
        / "ablations"
        / name
    )


def build_pipeline(
    config_name: str,
) -> C3Pipeline:
    config = load_ablation_config(
        config_name
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
            ROOT / "prompts" / "answer_prompt.txt"
        ),
    )


@pytest.mark.parametrize(
    ("config_name", "expected"),
    CONFIG_CASES.items(),
)
def test_each_config_enables_exactly_one_ablation(
    config_name: str,
    expected: tuple[str, str],
) -> None:
    expected_variant, expected_flag = expected

    settings = AblationSettings.from_config(
        load_ablation_config(config_name)
    )

    assert settings.variant == expected_variant
    assert settings.enabled_flags() == [
        expected_flag
    ]


def test_no_route_planner_forces_all_memory_types() -> None:
    route = force_all_memory_route()

    assert route.selected_types == list(
        MemoryType
    )
    assert all(
        route.scores[memory_type.value]
        == pytest.approx(1.0)
        for memory_type in MemoryType
    )


def test_ranked_top_k_preserves_rank_order_and_limits() -> None:
    config = load_ablation_config(
        "rc85_no_evidence_selector.yaml"
    )

    candidates = [
        MemoryCandidate(
            memory_id=f"s_{index}",
            memory_type=MemoryType.SEMANTIC,
            text=f"candidate evidence {index}",
            user_id="user01",
            final_score=1.0 - index / 10,
        )
        for index in range(7)
    ]

    selected = select_ranked_top_k(
        candidates=candidates,
        config=config,
    )

    assert selected
    assert [
        item.memory_id
        for item in selected
    ] == [
        item.memory_id
        for item in candidates[: len(selected)]
    ]

    assert len(selected) <= int(
        config["selection"]["max_evidence"]
    )

    assert all(
        item.metadata["selector_reason"]
        == "ablation_ranked_top_k"
        for item in selected
    )


def test_no_route_planner_pipeline_routes_all_types() -> None:
    pipeline = build_pipeline(
        "rc85_no_route_planner.yaml"
    )

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What is my current MSc "
                    "project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert set(
        result.selected_memory_types
    ) == {
        "episodic",
        "semantic",
        "procedural",
    }
    assert (
        result.debug["ablation_variant"]
        == "no_route_planner"
    )


def test_no_conflict_pipeline_skips_conflict_groups() -> None:
    pipeline = build_pipeline(
        "rc85_no_conflict_handling.yaml"
    )

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

    assert result.conflict_groups == []
    assert (
        result.debug[
            "conflict_handling_enabled"
        ]
        is False
    )


def test_no_gate_pipeline_marks_gate_bypass() -> None:
    pipeline = build_pipeline(
        "rc85_no_coverage_confidence_gate.yaml"
    )

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What is my current MSc "
                    "project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert (
        result.debug["confidence_gate_enabled"]
        is False
    )
    assert (
        result.debug["confidence_components"][
            "ablation_gate_bypassed"
        ]
        == pytest.approx(1.0)
    )


def test_no_selector_pipeline_uses_ranked_top_k() -> None:
    pipeline = build_pipeline(
        "rc85_no_evidence_selector.yaml"
    )

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "What is my current MSc "
                    "project scope?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert result.selected_evidence
    assert (
        result.debug["selection_mode"]
        == "ranked_top_k"
    )
    assert all(
        item.metadata.get("selector_reason")
        == "ablation_ranked_top_k"
        for item in result.selected_evidence
    )
