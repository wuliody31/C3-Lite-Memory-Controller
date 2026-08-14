from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.coverage_estimator import (
    CoverageEstimator,
)
from src.information_need_gain_v3 import (
    InformationNeedGainEvaluator,
)
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluator():
    config = load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )

    return InformationNeedGainEvaluator(
        CoverageEstimator(
            config
        )
    )


def candidate(
    memory_id: str,
    text: str,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=(
            MemoryType.SEMANTIC
        ),
        text=text,
        user_id="user01",
        status="current",
        confidence=1.0,
        final_score=0.8,
    )


def test_new_information_need_has_positive_gain():
    model = evaluator()

    selected = [
        candidate(
            "scope",
            (
                "The current project scope is "
                "a small memory controller prototype."
            ),
        )
    ]

    extra = candidate(
        "focus",
        (
            "The current project focus is "
            "controlled memory selection."
        ),
    )

    result = model.evaluate(
        candidate=extra,
        selected=selected,
        information_needs=[
            "current project scope",
            "controlled memory selection focus",
        ],
    )

    assert result.gain > 0.0

    assert (
        "controlled memory selection focus"
        in result.gained_needs
    )


def test_redundant_candidate_has_zero_gain():
    model = evaluator()

    selected = [
        candidate(
            "a",
            (
                "The current project focus is "
                "controlled memory selection."
            ),
        )
    ]

    redundant = candidate(
        "b",
        (
            "Controlled memory selection is "
            "the current project focus."
        ),
    )

    result = model.evaluate(
        candidate=redundant,
        selected=selected,
        information_needs=[
            "current project focus",
        ],
    )

    assert result.gain == 0.0
    assert result.gained_indices == []


def test_candidate_can_cover_multiple_uncovered_needs():
    model = evaluator()

    evidence = candidate(
        "multi",
        (
            "The project uses episodic memory "
            "for events and semantic memory "
            "for current facts."
        ),
    )

    result = model.evaluate(
        candidate=evidence,
        selected=[],
        information_needs=[
            "episodic memory events",
            "semantic memory current facts",
        ],
    )

    assert result.coverage_after == 1.0

    assert set(
        result.gained_indices
    ) == {
        0,
        1,
    }


def test_already_covered_need_does_not_gain_again():
    model = evaluator()

    selected = [
        candidate(
            "historical_1",
            (
                "The user originally wanted "
                "a broad memory chatbot."
            ),
        )
    ]

    second = candidate(
        "historical_2",
        (
            "Earlier the user considered "
            "a broad chatbot."
        ),
    )

    result = model.evaluate(
        candidate=second,
        selected=selected,
        information_needs=[
            "original historical project",
        ],
    )

    assert result.gain == 0.0

def test_set_coverage_distinguishes_scope_and_focus():
    model = evaluator()

    scope = candidate(
        "scope",
        (
            "The current project scope is "
            "a small memory controller prototype."
        ),
    )

    result = model.coverage_ratio(
        selected=[scope],
        information_needs=[
            "current project scope",
            "controlled memory selection focus",
        ],
    )

    assert result == 0.5


def test_set_coverage_reaches_one_with_distinct_evidence():
    model = evaluator()

    scope = candidate(
        "scope",
        (
            "The current project scope is "
            "a small memory controller prototype."
        ),
    )

    focus = candidate(
        "focus",
        (
            "The current project focus is "
            "controlled memory selection."
        ),
    )

    result = model.coverage_ratio(
        selected=[
            scope,
            focus,
        ],
        information_needs=[
            "current project scope",
            "controlled memory selection focus",
        ],
    )

    assert result == 1.0