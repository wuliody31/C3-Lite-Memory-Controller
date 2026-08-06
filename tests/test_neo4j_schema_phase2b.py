from __future__ import annotations

from typing import Any

from src.neo4j_schema import Neo4jSchemaBootstrap


class FakeResult(list):
    def __init__(
        self,
        rows=(),
    ) -> None:
        super().__init__(rows)
        self.consumed = False

    def consume(self):
        self.consumed = True
        return None


class FakeSession:
    def __init__(
        self,
        driver: "FakeDriver",
    ) -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    def run(
        self,
        cypher: str,
        **params: Any,
    ) -> FakeResult:
        self.driver.calls.append(
            (cypher, params)
        )

        if "SHOW CONSTRAINTS" in cypher:
            return FakeResult(
                self.driver.constraint_rows
            )

        if "SHOW INDEXES" in cypher:
            return FakeResult(
                self.driver.index_rows
            )

        result = FakeResult()
        self.driver.results.append(result)
        return result


class FakeDriver:
    def __init__(self) -> None:
        self.calls = []
        self.results = []
        self.session_calls = []
        self.constraint_rows = []
        self.index_rows = []

    def session(self, **kwargs):
        self.session_calls.append(kwargs)
        return FakeSession(self)


def test_schema_bootstrap_is_idempotent():
    driver = FakeDriver()

    bootstrap = Neo4jSchemaBootstrap(
        driver=driver,
        database="neo4j",
    )

    result = bootstrap.apply(
        wait_seconds=30
    )

    assert len(result["applied"]) == 6
    assert len(driver.calls) == 7

    statements = "\n".join(
        cypher
        for cypher, _ in driver.calls
    )

    assert (
        "CREATE CONSTRAINT "
        "c3_semantic_fact_id_unique"
        in statements
    )
    assert (
        "CREATE CONSTRAINT "
        "c3_episode_id_unique"
        in statements
    )
    assert (
        "CREATE FULLTEXT INDEX "
        "semantic_fact_text"
        in statements
    )
    assert (
        "CREATE FULLTEXT INDEX episode_text"
        in statements
    )
    assert statements.count(
        "IF NOT EXISTS"
    ) == 6

    assert driver.session_calls == [
        {"database": "neo4j"}
    ]


def test_bootstrap_waits_for_indexes():
    driver = FakeDriver()

    bootstrap = Neo4jSchemaBootstrap(
        driver=driver
    )

    bootstrap.apply(
        wait_seconds=45
    )

    cypher, params = driver.calls[-1]

    assert "db.awaitIndexes" in cypher
    assert params == {
        "wait_seconds": 45
    }


def test_schema_inspection_maps_rows():
    driver = FakeDriver()

    driver.constraint_rows = [
        {
            "name": (
                "c3_semantic_fact_id_unique"
            ),
            "type": "UNIQUENESS",
        }
    ]

    driver.index_rows = [
        {
            "name": "semantic_fact_text",
            "type": "FULLTEXT",
            "state": "ONLINE",
        }
    ]

    bootstrap = Neo4jSchemaBootstrap(
        driver=driver
    )

    result = bootstrap.inspect()

    assert result["constraints"][0]["name"] == (
        "c3_semantic_fact_id_unique"
    )

    assert result["indexes"][0] == {
        "name": "semantic_fact_text",
        "type": "FULLTEXT",
        "state": "ONLINE",
    }
