from __future__ import annotations

from datetime import datetime, timezone

from src.retrievers.neo4j_store import (
    Neo4jMemoryStore,
)
from src.schemas import (
    MemoryType,
    QueryFeatures,
    QueryMode,
    QueryState,
)


def features(
    mode: QueryMode,
) -> QueryFeatures:
    historical = (
        mode == QueryMode.HISTORICAL
    )

    timeline = (
        mode == QueryMode.TIMELINE
    )

    return QueryFeatures(
        normalised_query=(
            "where did the user live"
        ),
        tokens=[
            "where",
            "user",
            "live",
        ],
        entities=["Nottingham"],
        temporal_expressions=[],
        query_mode=mode,
        task_type=None,
        asks_current_state=(
            mode == QueryMode.CURRENT
        ),
        asks_historical_state=historical,
        asks_timeline=timeline,
        asks_procedure=False,
        asks_explanation=False,
        asks_conflict=False,
        information_needs=[],
    )


def semantic_row(
    memory_id: str,
    value: str,
    status: str,
    *,
    relations=None,
):
    return {
        "node": {
            "id": memory_id,
            "fact_id": memory_id,
            "user_id": "user_01",
            "text": f"User lives in {value}.",
            "subject": "user_01",
            "predicate": "user.residence",
            "object": value,
            "status": status,
            "valid_from": (
                "2026-07-15T00:00:00+00:00"
            ),
            "confidence": 1.0,
        },
        "score": 1.0,
        "relations": relations or [],
    }


class CapturingNeo4jStore(
    Neo4jMemoryStore
):
    def __init__(self):
        self.semantic_index = (
            "semantic_fact_text"
        )
        self.episode_index = "episode_text"
        self.database = None
        self._owns_driver = False
        self.calls = []

    def _run(self, cypher, **params):
        self.calls.append(
            (cypher, params)
        )

        new_row = semantic_row(
            "state_london",
            "London",
            "current",
            relations=[
                {
                    "type": "SUPERSEDES",
                    "target_id": (
                        "state_nottingham"
                    ),
                }
            ],
        )

        if params.get(
            "include_archived"
        ):
            return [
                new_row,
                semantic_row(
                    "state_nottingham",
                    "Nottingham",
                    "superseded",
                ),
            ]

        return [new_row]


def test_current_query_excludes_superseded():
    store = CapturingNeo4jStore()

    result = store.retrieve(
        memory_type=MemoryType.SEMANTIC,
        state=QueryState(
            query="Where do I live?",
            user_id="user_01",
        ),
        features=features(
            QueryMode.CURRENT
        ),
        top_k=10,
    )

    assert [
        item.memory_id
        for item in result
    ] == ["state_london"]

    _, params = store.calls[-1]

    assert (
        params["include_archived"]
        is False
    )


def test_historical_query_includes_superseded():
    store = CapturingNeo4jStore()

    result = store.retrieve(
        memory_type=MemoryType.SEMANTIC,
        state=QueryState(
            query="Where did I live before?",
            user_id="user_01",
        ),
        features=features(
            QueryMode.HISTORICAL
        ),
        top_k=10,
    )

    assert {
        item.memory_id
        for item in result
    } == {
        "state_london",
        "state_nottingham",
    }

    _, params = store.calls[-1]

    assert (
        params["include_archived"]
        is True
    )


def test_timeline_query_expands_history():
    assert (
        Neo4jMemoryStore._include_archived(
            features=features(
                QueryMode.TIMELINE
            ),
            requested=False,
        )
        is True
    )


def test_explicit_include_archived_overrides_mode():
    assert (
        Neo4jMemoryStore._include_archived(
            features=features(
                QueryMode.CURRENT
            ),
            requested=True,
        )
        is True
    )


def test_supersedes_relation_is_preserved():
    store = CapturingNeo4jStore()

    result = store.retrieve(
        memory_type=MemoryType.SEMANTIC,
        state=QueryState(
            query="Where do I live?",
            user_id="user_01",
            current_time=datetime(
                2026,
                8,
                6,
                tzinfo=timezone.utc,
            ),
        ),
        features=features(
            QueryMode.CURRENT
        ),
        top_k=10,
    )

    assert result[0].relations == [
        {
            "type": "SUPERSEDES",
            "target_id": "state_nottingham",
        }
    ]


def test_fulltext_parameter_avoids_reserved_query_name():
    """Do not collide with Session.run(query=...)."""

    store = CapturingNeo4jStore()

    store.retrieve(
        memory_type=MemoryType.SEMANTIC,
        state=QueryState(
            query="Where do I live?",
            user_id="user_01",
        ),
        features=features(
            QueryMode.CURRENT
        ),
        top_k=10,
    )

    _, params = store.calls[-1]

    assert "query" not in params
    assert params["search_query"]

