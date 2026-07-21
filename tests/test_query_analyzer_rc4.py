from __future__ import annotations

import pytest

from src.query_analyzer import QueryAnalyzer
from src.schemas import QueryMode


@pytest.fixture()
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer(
        {
            "query_analysis": {
                "stopwords": [],
                "aliases": {},
            }
        }
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "What are the three memory types in my project and what does each record?",
            {
                "task_type": None,
                "procedure": False,
                "historical": False,
            },
        ),
        (
            "Did I decide to train my own model for this project?",
            {
                "task_type": None,
                "procedure": False,
                "historical": True,
            },
        ),
        (
            "What baselines have I selected for the project evaluation?",
            {
                "procedure": False,
                "historical": True,
                "current": True,
                "mode": QueryMode.TIMELINE,
            },
        ),
        (
            "Should I claim production deployment in my CV if there is no evidence?",
            {
                "procedure": True,
                "explanation": False,
            },
        ),
        (
            "What should I say if there is no stored evidence for a project claim?",
            {
                "procedure": True,
                "explanation": False,
            },
        ),
        (
            "Is my project currently a large multi-agent system or a smaller C3-Lite controller prototype?",
            {
                "procedure": False,
                "conflict": True,
                "historical": True,
                "current": True,
                "mode": QueryMode.TIMELINE,
            },
        ),
        (
            "Should my evaluation mainly be a user study or a controlled fictional dataset with baseline comparison?",
            {
                "procedure": True,
                "conflict": True,
            },
        ),
        (
            "Please explain why my dataset should not be centred on Neo4j.",
            {
                "procedure": True,
                "explanation": True,
            },
        ),
        (
            "What evidence supports using memory explanations as an evaluation dimension?",
            {
                "procedure": True,
                "explanation": True,
            },
        ),
    ],
)
def test_rc4_query_analysis(
    analyzer: QueryAnalyzer,
    query: str,
    expected: dict[str, object],
) -> None:
    features = analyzer.analyse(query)

    if "task_type" in expected:
        assert features.task_type == expected["task_type"]
    if "procedure" in expected:
        assert features.asks_procedure is expected["procedure"]
    if "explanation" in expected:
        assert features.asks_explanation is expected["explanation"]
    if "conflict" in expected:
        assert features.asks_conflict is expected["conflict"]
    if "historical" in expected:
        assert features.asks_historical_state is expected["historical"]
    if "current" in expected:
        assert features.asks_current_state is expected["current"]
    if "mode" in expected:
        assert features.query_mode == expected["mode"]
