from __future__ import annotations

from pathlib import Path
from unittest import result

import pytest

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import InMemoryMemoryStore, ProceduralJsonStore
from src.schemas import QueryState


ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> C3Pipeline:
    """Build a deterministic JSON + MockBackbone regression pipeline."""

    config = load_config(
        ROOT / "configs" / "c3_lite_v2_2_final.yaml"
    )
    stopwords = set(config["query_analysis"]["stopwords"])

    memory_store = InMemoryMemoryStore.from_json_file(
        str(ROOT / "examples" / "demo_memories.json"),
        stopwords,
    )
    procedure_store = ProceduralJsonStore.from_file(
        str(ROOT / "examples" / "demo_procedures.json"),
        stopwords,
    )

    return C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=MockBackbone(),
        prompt_template=ROOT / "prompts" / "answer_prompt.txt",
    )


def test_timeline_entities_conflict_roles_and_coverage() -> None:
    pipeline = build_pipeline()
    try:
        result = pipeline.answer(
            QueryState(
                query="How did my project scope change over time?",
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert result.query_mode.value == "timeline"
    assert result.selected_memory_types == ["episodic", "semantic"]

    entities = result.debug["entities"]
    assert "How" not in entities
    assert "timeline" not in entities
    assert "dissertation" not in entities
    assert "project scope" in entities

    assert len(result.conflict_groups) == 1
    conflict = result.conflict_groups[0]
    assert conflict.preferred_ids == ["s_user01_003"]
    assert conflict.historical_ids == ["s_user01_004"]
    assert conflict.unresolved is False

    evidence_by_id = {
        item.memory_id: item
        for item in result.selected_evidence
    }
    assert (
        evidence_by_id["s_user01_004"].resolution_action
        == "historical"
    )
    assert (
        evidence_by_id["s_user01_003"].resolution_action
        == "current_endpoint"
    )

    assert result.coverage == pytest.approx(1.0)
    assert result.decision.value == "direct"


def test_current_state_selects_only_current_semantic_fact() -> None:
    pipeline = build_pipeline()
    try:
        result = pipeline.answer(
            QueryState(
                query="What is my current MSc project scope?",
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert result.query_mode.value == "current_state"
    assert result.selected_memory_types == ["semantic"]
    assert result.selected_ids == ["s_user01_003"]
    assert result.coverage == pytest.approx(1.0)
    assert result.decision.value == "direct"
    assert "s_user01_003" in result.answer
    assert "s_user01_004" not in result.answer


def test_procedural_rule_is_sufficient_and_mock_reads_rc3_header() -> None:
    pipeline = build_pipeline()
    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "When discussing my dissertation with my supervisor, "
                    "how should the answer be written?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert result.query_mode.value == "procedural"
    assert result.selected_memory_types == [
        "semantic",
        "procedural",
    ]

    assert "p_user01_001" in result.selected_ids
    assert result.coverage == pytest.approx(1.0)
    assert result.decision.value == "direct"

    # Guards against the old MockBackbone parser that required "] score=".
    assert "p_user01_001" in result.answer
    assert "not enough stored evidence" not in result.answer.lower()


def test_unsupported_preference_query_abstains() -> None:
    pipeline = build_pipeline()
    try:
        result = pipeline.answer(
            QueryState(
                query="What is my favourite programming language?",
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    assert result.selected_ids == []
    assert result.selected_evidence == []
    assert result.coverage == pytest.approx(0.0)
    assert result.decision.value == "abstain"
    assert "not enough stored evidence" in result.answer.lower()
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_llm_facing_prompt_hides_internal_scores_and_is_well_formed() -> None:
    pipeline = build_pipeline()
    try:
        result = pipeline.answer(
            QueryState(
                query="What is my current MSc project scope?",
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    prompt = result.final_prompt

    assert prompt.splitlines().count("SYSTEM") == 1
    assert "score=" not in prompt
    assert "status=current" not in prompt
    assert "route_score" not in prompt
    assert "inventunsupported" not in prompt
    assert "Howdid" not in prompt
    assert "EPISODICEVIDENCE" not in prompt
    assert "Answerdirectly" not in prompt
    assert "a technical explanation" in prompt


def test_rc5_full_candidate_trace_covers_raw_candidates() -> None:
    pipeline = build_pipeline()

    try:
        result = pipeline.answer(
            QueryState(
                query=(
                    "When discussing my dissertation with my supervisor, "
                    "how should the answer be written?"
                ),
                user_id="user01",
            )
        )
    finally:
        pipeline.close()

    trace = result.debug["full_candidate_score_trace"]

    assert len(trace) == result.debug[
        "candidate_count_raw_retrieved"
    ]

    trace_ids = {
        row["memory_id"]
        for row in trace
    }

    assert trace_ids == set(result.raw_retrieved_ids)

    required_fields = {
        "memory_id",
        "memory_type",
        "candidate_utility_v0",
        "post_resolution_final_score",
        "rank_global_all_scored",
        "rank_within_type_all_scored",
        "passed_ranker_gate",
        "kept_after_type_top_k",
        "survived_conflict_resolution",
        "selected_final",
        "estimated_token_cost",
        "utility_per_estimated_token",
        "drop_stage",
    }

    for row in trace:
        assert required_fields <= row.keys()

        assert row["drop_stage"] in {
            None,
            "ranker_gate",
            "type_top_k",
            "conflict_resolution",
            "evidence_selector",
        }

        assert row["estimated_token_cost"] >= 1

    selected_from_trace = {
        row["memory_id"]
        for row in trace
        if row["selected_final"]
    }

    assert selected_from_trace == set(result.selected_ids)