from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.errors import RetrievalError
from src.retrievers.neo4j_store import Neo4jMemoryStore
from src.schemas import (
    MemoryType,
    QueryFeatures,
    QueryState,
)


class FakeSession:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.rows = rows or []
        self.error = error

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback

    def run(
        self,
        cypher: str,
        **params: Any,
    ) -> list[dict[str, Any]]:
        del cypher, params

        if self.error is not None:
            raise self.error

        return self.rows


class FakeDriver:
    def __init__(self, session: FakeSession) -> None:
        self.fake_session = session

    def session(self, **kwargs: Any) -> FakeSession:
        del kwargs
        return self.fake_session


def episodic_features() -> QueryFeatures:
    return cast(
        QueryFeatures,
        SimpleNamespace(
            normalised_query="project scope",
        ),
    )


def test_backend_query_failure_becomes_retrieval_error() -> None:
    store = Neo4jMemoryStore(
        config={},
        driver=FakeDriver(
            FakeSession(
                error=OSError(
                    "simulated connection failure"
                )
            )
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="Neo4j query execution failed",
    ) as captured:
        store.retrieve(
            memory_type=MemoryType.EPISODIC,
            state=QueryState(
                "What is my project scope?",
                "user01",
            ),
            features=episodic_features(),
            top_k=5,
        )

    assert isinstance(captured.value, RuntimeError)
    assert isinstance(
        captured.value.__cause__,
        OSError,
    )


def test_valid_empty_backend_result_is_not_failure() -> None:
    store = Neo4jMemoryStore(
        config={},
        driver=FakeDriver(
            FakeSession(rows=[])
        ),
    )

    result = store.retrieve(
        memory_type=MemoryType.EPISODIC,
        state=QueryState(
            "What is my project scope?",
            "user01",
        ),
        features=episodic_features(),
        top_k=5,
    )

    assert result == []
