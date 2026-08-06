from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.lifecycle import (
    LifecycleRecord,
    LifecycleStoreError,
    MemoryStatus,
)
from src.lifecycle_neo4j import Neo4jLifecycleStore
from src.schemas import MemoryType


def utc(
    year: int,
    month: int,
    day: int,
) -> datetime:
    return datetime(
        year,
        month,
        day,
        tzinfo=timezone.utc,
    )


class FakeResult(list):
    pass


class FakeTransaction:
    def __init__(self) -> None:
        self.calls: list[
            tuple[str, dict[str, Any]]
        ] = []
        self.existing_add: list[str] = []
        self.existing_update: list[str] = []

    def run(
        self,
        cypher: str,
        **params: Any,
    ) -> FakeResult:
        self.calls.append(
            (cypher, params)
        )

        if "lifecycle_precheck_add" in cypher:
            return FakeResult(
                [
                    {"memory_id": memory_id}
                    for memory_id
                    in self.existing_add
                ]
            )

        if "lifecycle_precheck_update" in cypher:
            return FakeResult(
                [
                    {"memory_id": memory_id}
                    for memory_id
                    in self.existing_update
                ]
            )

        return FakeResult()


class FakeSession:
    def __init__(
        self,
        driver: "FakeDriver",
    ) -> None:
        self.driver = driver
        self.tx = driver.tx

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> bool:
        return False

    def execute_write(
        self,
        callback,
        *args,
    ):
        self.driver.write_transactions += 1
        return callback(
            self.tx,
            *args,
        )

    def run(
        self,
        cypher: str,
        **params: Any,
    ) -> FakeResult:
        self.driver.read_calls.append(
            (cypher, params)
        )

        return FakeResult(
            self.driver.read_rows
        )


class FakeDriver:
    def __init__(self) -> None:
        self.tx = FakeTransaction()
        self.write_transactions = 0
        self.session_calls: list[
            dict[str, Any]
        ] = []
        self.read_calls: list[
            tuple[str, dict[str, Any]]
        ] = []
        self.read_rows: list[
            dict[str, Any]
        ] = []
        self.closed = False

    def session(
        self,
        **kwargs: Any,
    ) -> FakeSession:
        self.session_calls.append(kwargs)
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True


def semantic(
    *,
    memory_id: str,
    value: str,
    status: MemoryStatus,
    valid_from: datetime,
    valid_to: datetime | None = None,
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> LifecycleRecord:
    return LifecycleRecord(
        memory_id=memory_id,
        user_id="user_01",
        memory_type=MemoryType.SEMANTIC,
        text=f"User lives in {value}.",
        status=status,
        valid_from=valid_from,
        state_key="user.residence",
        value=value,
        valid_to=valid_to,
        source_turn_id=f"turn_{memory_id}",
        supersedes=supersedes,
        superseded_by=superseded_by,
        metadata={
            "lifecycle_role": "semantic_state",
            "authority": "user_confirmed",
        },
    )


def episode(
    *,
    memory_id: str,
    old_id: str,
    new_id: str,
) -> LifecycleRecord:
    return LifecycleRecord(
        memory_id=memory_id,
        user_id="user_01",
        memory_type=MemoryType.EPISODIC,
        text=(
            "User moved from Nottingham "
            "to London."
        ),
        status=MemoryStatus.CURRENT,
        valid_from=utc(2026, 7, 15),
        state_key="user.residence",
        source_turn_id="turn_move",
        metadata={
            "lifecycle_role": "state_transition",
            "old_memory_id": old_id,
            "new_memory_id": new_id,
            "old_value": "Nottingham",
            "new_value": "London",
        },
    )


def find_call(
    driver: FakeDriver,
    marker: str,
) -> tuple[str, dict[str, Any]]:
    for cypher, params in driver.tx.calls:
        if marker in cypher:
            return cypher, params

    raise AssertionError(
        f"No transaction call contains {marker!r}"
    )


def test_add_initial_semantic_uses_one_transaction():
    driver = FakeDriver()
    store = Neo4jLifecycleStore(
        driver=driver
    )

    record = semantic(
        memory_id="state_nottingham",
        value="Nottingham",
        status=MemoryStatus.CURRENT,
        valid_from=utc(2026, 6, 1),
    )

    store.commit(add=[record])

    assert driver.write_transactions == 1

    _, params = find_call(
        driver,
        "lifecycle_add_semantic",
    )

    row = params["rows"][0]
    props = row["properties"]

    assert row["memory_id"] == "state_nottingham"
    assert props["fact_id"] == "state_nottingham"
    assert props["status"] == "current"
    assert props["state_key"] == "user.residence"
    assert props["object"] == "Nottingham"
    assert props["lifecycle_managed"] is True


def test_supersession_is_written_atomically():
    driver = FakeDriver()
    driver.tx.existing_update = [
        "state_nottingham"
    ]

    store = Neo4jLifecycleStore(
        driver=driver
    )

    old_record = semantic(
        memory_id="state_nottingham",
        value="Nottingham",
        status=MemoryStatus.SUPERSEDED,
        valid_from=utc(2026, 6, 1),
        valid_to=utc(2026, 7, 15),
        superseded_by="state_london",
    )

    new_record = semantic(
        memory_id="state_london",
        value="London",
        status=MemoryStatus.CURRENT,
        valid_from=utc(2026, 7, 15),
        supersedes="state_nottingham",
    )

    event = episode(
        memory_id="episode_move",
        old_id="state_nottingham",
        new_id="state_london",
    )

    store.commit(
        update=[old_record],
        add=[new_record, event],
    )

    assert driver.write_transactions == 1

    find_call(
        driver,
        "lifecycle_update_semantic",
    )
    find_call(
        driver,
        "lifecycle_add_semantic",
    )
    find_call(
        driver,
        "lifecycle_add_episodic",
    )

    supersedes_cypher, supersedes_params = (
        find_call(
            driver,
            "lifecycle_create_supersedes",
        )
    )

    assert (
        "MERGE (new)-[:SUPERSEDES]->(old)"
        in supersedes_cypher
    )

    assert supersedes_params["rows"] == [
        {
            "new_memory_id": "state_london",
            "old_memory_id": "state_nottingham",
        }
    ]

    episode_cypher, episode_params = find_call(
        driver,
        "lifecycle_create_episode_links",
    )

    assert (
        "MERGE (event)-[:FROM_STATE]->(old)"
        in episode_cypher
    )
    assert (
        "MERGE (event)-[:TO_STATE]->(new)"
        in episode_cypher
    )

    assert episode_params["rows"] == [
        {
            "episode_memory_id": "episode_move",
            "old_memory_id": "state_nottingham",
            "new_memory_id": "state_london",
        }
    ]


def test_add_existing_memory_is_rejected():
    driver = FakeDriver()
    driver.tx.existing_add = [
        "state_nottingham"
    ]

    store = Neo4jLifecycleStore(
        driver=driver
    )

    record = semantic(
        memory_id="state_nottingham",
        value="Nottingham",
        status=MemoryStatus.CURRENT,
        valid_from=utc(2026, 6, 1),
    )

    with pytest.raises(
        LifecycleStoreError,
        match="existing lifecycle memory IDs",
    ):
        store.commit(add=[record])

    assert not any(
        "lifecycle_add_semantic" in cypher
        for cypher, _ in driver.tx.calls
    )


def test_update_missing_memory_is_rejected():
    driver = FakeDriver()
    store = Neo4jLifecycleStore(
        driver=driver
    )

    record = semantic(
        memory_id="missing_state",
        value="Nottingham",
        status=MemoryStatus.SUPERSEDED,
        valid_from=utc(2026, 6, 1),
        valid_to=utc(2026, 7, 15),
        superseded_by="state_london",
    )

    with pytest.raises(
        LifecycleStoreError,
        match="missing lifecycle memory IDs",
    ):
        store.commit(update=[record])

    assert not any(
        "lifecycle_update_semantic" in cypher
        for cypher, _ in driver.tx.calls
    )


def test_get_maps_semantic_fact():
    driver = FakeDriver()

    driver.read_rows = [
        {
            "labels": ["SemanticFact"],
            "properties": {
                "id": "state_london",
                "fact_id": "state_london",
                "user_id": "user_01",
                "text": (
                    "User currently lives in London."
                ),
                "status": "current",
                "state_key": "user.residence",
                "value": "London",
                "valid_from": (
                    "2026-07-15T00:00:00+00:00"
                ),
                "source_turn_id": "turn_move",
                "supersedes": "state_nottingham",
                "lifecycle_managed": True,
                "metadata_json": (
                    '{"lifecycle_role":'
                    '"semantic_state"}'
                ),
            },
        }
    ]

    store = Neo4jLifecycleStore(
        driver=driver
    )

    record = store.get("state_london")

    assert record is not None
    assert (
        record.memory_type
        == MemoryType.SEMANTIC
    )
    assert record.value == "London"
    assert record.status == MemoryStatus.CURRENT
    assert (
        record.supersedes
        == "state_nottingham"
    )
    assert (
        record.metadata["lifecycle_role"]
        == "semantic_state"
    )


def test_list_passes_lifecycle_filters():
    driver = FakeDriver()

    driver.read_rows = [
        {
            "labels": ["Episode"],
            "properties": {
                "id": "episode_move",
                "episode_id": "episode_move",
                "user_id": "user_01",
                "text": (
                    "User moved from Nottingham "
                    "to London."
                ),
                "status": "current",
                "state_key": "user.residence",
                "timestamp": (
                    "2026-07-15T00:00:00+00:00"
                ),
                "source_turn_id": "turn_move",
                "lifecycle_managed": True,
                "metadata_json": (
                    '{"lifecycle_role":'
                    '"state_transition"}'
                ),
            },
        }
    ]

    store = Neo4jLifecycleStore(
        driver=driver,
        database="neo4j",
    )

    records = store.list(
        user_id="user_01",
        memory_type=MemoryType.EPISODIC,
        state_key="user.residence",
        status=MemoryStatus.CURRENT,
    )

    assert len(records) == 1
    assert (
        records[0].memory_type
        == MemoryType.EPISODIC
    )

    assert driver.session_calls[-1] == {
        "database": "neo4j"
    }

    _, params = driver.read_calls[-1]

    assert params == {
        "user_id": "user_01",
        "memory_type": "episodic",
        "state_key": "user.residence",
        "status": "current",
    }


def test_metadata_is_json_encoded():
    record = episode(
        memory_id="episode_move",
        old_id="state_nottingham",
        new_id="state_london",
    )

    properties = (
        Neo4jLifecycleStore._properties(record)
    )

    assert isinstance(
        properties["metadata_json"],
        str,
    )
    assert (
        '"old_memory_id": "state_nottingham"'
        in properties["metadata_json"]
    )


def test_same_id_cannot_be_added_and_updated():
    driver = FakeDriver()
    store = Neo4jLifecycleStore(
        driver=driver
    )

    record = semantic(
        memory_id="state_x",
        value="London",
        status=MemoryStatus.CURRENT,
        valid_from=utc(2026, 7, 15),
    )

    with pytest.raises(
        LifecycleStoreError,
        match="added and updated",
    ):
        store.commit(
            add=[record],
            update=[record],
        )

    assert driver.write_transactions == 0


def test_injected_driver_is_not_closed():
    driver = FakeDriver()
    store = Neo4jLifecycleStore(
        driver=driver
    )

    store.close()

    assert driver.closed is False
