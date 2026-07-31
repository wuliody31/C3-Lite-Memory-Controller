from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.coverage_estimator import CoverageEstimator
from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner
from src.schemas import MemoryCandidate, MemoryType


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return load_config(
        ROOT / "configs" / "c3_lite_v2_2_final.yaml"
    )


def candidate(
    memory_id: str,
    text: str,
    *,
    memory_type: MemoryType,
    metadata: dict | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=memory_type,
        text=text,
        user_id="user01",
        status="current",
        confidence=1.0,
        lexical_score=0.0,
        graph_entity_score=0.0,
        final_score=0.8,
        metadata=metadata or {},
    )


def test_factual_current_comparison_does_not_force_procedure() -> None:
    cfg = config()
    features = QueryAnalyzer(cfg).analyse(
        "Which is more current: Data Analyst as the main "
        "direction or LLM Agent Engineer as the main direction?"
    )

    decision = RoutePlanner(cfg).plan(features)

    assert MemoryType.EPISODIC in decision.selected_types
    assert MemoryType.SEMANTIC in decision.selected_types
    assert MemoryType.PROCEDURAL not in decision.selected_types


def test_historical_wording_counts_as_historical_coverage() -> None:
    estimator = CoverageEstimator(config())

    evidence = candidate(
        "e_original",
        "The user initially wanted to build a broad "
        "long-term memory chatbot.",
        memory_type=MemoryType.EPISODIC,
    )

    coverage = estimator.compute(
        [
            "originally want build before narrowing scope",
            "historical event or state",
        ],
        [evidence],
    )

    assert coverage == pytest.approx(1.0)


def test_preliminary_procedural_match_is_applicable() -> None:
    estimator = CoverageEstimator(config())

    evidence = candidate(
        "p_no_evidence",
        "When no evidence is stored for a claim, say that "
        "there is not enough evidence and do not guess.",
        memory_type=MemoryType.PROCEDURAL,
        metadata={
            "preliminary_procedural_score": 0.75,
            "direct_match": False,
            "task_match": 0.60,
        },
    )

    coverage = estimator.compute(
        [
            "should say no stored evidence project claim",
            "applicable procedure or rule",
        ],
        [evidence],
    )

    assert coverage == pytest.approx(1.0)


def test_three_memory_definitions_have_complete_coverage() -> None:
    estimator = CoverageEstimator(config())

    selected = [
        candidate(
            "s_ep",
            "episodic memory records what happened and when",
            memory_type=MemoryType.SEMANTIC,
        ),
        candidate(
            "s_sem",
            "semantic memory records currently believed facts",
            memory_type=MemoryType.SEMANTIC,
        ),
        candidate(
            "s_proc",
            "procedural memory records rules and response policies",
            memory_type=MemoryType.SEMANTIC,
        ),
    ]

    coverage = estimator.compute(
        [
            "three memory types project",
            "what each record",
        ],
        selected,
    )

    assert coverage == pytest.approx(1.0)


def test_scope_backend_evidence_covers_scope_validation() -> None:
    estimator = CoverageEstimator(config())

    evidence = candidate(
        "s_neo4j_role",
        "Neo4j role is implementation backend not research core.",
        memory_type=MemoryType.SEMANTIC,
    )

    coverage = estimator.compute(
        ["project mainly proving neo4j useful"],
        [evidence],
    )

    assert coverage == pytest.approx(1.0)
