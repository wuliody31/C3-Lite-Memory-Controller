from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from src.candidate_budget import BoundaryAwareCandidateBudget
from src.schemas import MemoryType


def make_candidate(
    memory_id: str,
    score: float,
    memory_type: MemoryType = MemoryType.SEMANTIC,
):
    return SimpleNamespace(
        memory_id=memory_id,
        memory_type=memory_type,
        final_score=score,
        confidence=1.0,
        timestamp=datetime(2026, 1, 1),
        metadata={},
    )


def config(
    *,
    epsilon: float = 0.01,
    max_extra_per_type: int = 5,
    max_total_extra: int = 8,
):
    return {
        "candidate_budget": {
            "enabled": True,
            "score_epsilon": epsilon,
            "max_extra_per_memory_type": max_extra_per_type,
            "max_total_extra": max_total_extra,
        }
    }


def top_k():
    return {
        "semantic": 3,
        "episodic": 3,
        "procedural": 2,
    }


def test_keeps_nominal_top_k() -> None:
    allocator = BoundaryAwareCandidateBudget(config(epsilon=0.0))
    candidates = [
        make_candidate(f"s_{index}", 1.0 - index * 0.1)
        for index in range(5)
    ]

    selected = allocator.select(
        ranked_pool=candidates,
        selected_types=[MemoryType.SEMANTIC],
        top_k=top_k(),
    )

    assert [item.memory_id for item in selected] == ["s_0", "s_1", "s_2"]


def test_preserves_exact_cutoff_ties() -> None:
    allocator = BoundaryAwareCandidateBudget(config(epsilon=0.0))
    candidates = [
        make_candidate("s_1", 0.70),
        make_candidate("s_2", 0.60),
        make_candidate("s_3", 0.50),
        make_candidate("s_4", 0.50),
        make_candidate("s_5", 0.40),
    ]

    selected = allocator.select(
        ranked_pool=candidates,
        selected_types=[MemoryType.SEMANTIC],
        top_k=top_k(),
    )

    assert [item.memory_id for item in selected] == [
        "s_1",
        "s_2",
        "s_3",
        "s_4",
    ]
    assert candidates[3].metadata["candidate_budget_reason"] == (
        "boundary_extension"
    )


def test_keeps_near_boundary_candidate_within_epsilon() -> None:
    allocator = BoundaryAwareCandidateBudget(config(epsilon=0.01))
    candidates = [
        make_candidate("s_1", 0.70),
        make_candidate("s_2", 0.60),
        make_candidate("s_3", 0.50),
        make_candidate("s_4", 0.493),
        make_candidate("s_5", 0.47),
    ]

    selected = allocator.select(
        ranked_pool=candidates,
        selected_types=[MemoryType.SEMANTIC],
        top_k=top_k(),
    )

    assert [item.memory_id for item in selected] == [
        "s_1",
        "s_2",
        "s_3",
        "s_4",
    ]


def test_rejects_candidate_outside_epsilon() -> None:
    allocator = BoundaryAwareCandidateBudget(config(epsilon=0.01))
    candidates = [
        make_candidate("s_1", 0.70),
        make_candidate("s_2", 0.60),
        make_candidate("s_3", 0.50),
        make_candidate("s_4", 0.48),
    ]

    selected = allocator.select(
        ranked_pool=candidates,
        selected_types=[MemoryType.SEMANTIC],
        top_k=top_k(),
    )

    assert [item.memory_id for item in selected] == ["s_1", "s_2", "s_3"]


def test_respects_global_extra_cap() -> None:
    allocator = BoundaryAwareCandidateBudget(
        config(
            epsilon=0.02,
            max_extra_per_type=5,
            max_total_extra=1,
        )
    )
    ranked_pool = [
        make_candidate("s_1", 0.60),
        make_candidate("e_1", 0.60, MemoryType.EPISODIC),
        make_candidate("s_2", 0.50),
        make_candidate("e_2", 0.50, MemoryType.EPISODIC),
        make_candidate("s_3", 0.49),
        make_candidate("e_3", 0.49, MemoryType.EPISODIC),
    ]

    selected = allocator.select(
        ranked_pool=ranked_pool,
        selected_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        top_k={
            "semantic": 2,
            "episodic": 2,
            "procedural": 2,
        },
    )

    assert len(selected) == 5
