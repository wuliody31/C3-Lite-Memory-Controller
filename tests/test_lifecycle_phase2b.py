from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.lifecycle import (
    InMemoryLifecycleStore,
    LifecycleInvariantError,
    LifecycleOperation,
    LifecycleRecord,
    LifecycleReplayConflict,
    LifecycleService,
    MemoryStatus,
)
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


def build_service() -> tuple[
    InMemoryLifecycleStore,
    LifecycleService,
]:
    store = InMemoryLifecycleStore()
    return store, LifecycleService(store)


def add_nottingham(
    service: LifecycleService,
):
    return service.ingest_state(
        user_id="user_01",
        state_key="user.residence",
        value="Nottingham",
        text="User lives in Nottingham.",
        observed_at=utc(2026, 6, 1),
        source_turn_id="turn_01",
    )


def move_to_london(
    service: LifecycleService,
):
    return service.ingest_state(
        user_id="user_01",
        state_key="user.residence",
        value="London",
        text="User currently lives in London.",
        transition_text=(
            "User moved from Nottingham "
            "to London in July 2026."
        ),
        observed_at=utc(2026, 7, 15),
        source_turn_id="turn_03",
    )


def test_initial_state_is_added_as_current():
    store, service = build_service()

    result = add_nottingham(service)

    assert (
        result.operation
        == LifecycleOperation.ADD_INITIAL
    )

    current = service.current_state(
        user_id="user_01",
        state_key="user.residence",
    )

    assert current is not None
    assert current.value == "Nottingham"
    assert current.status == MemoryStatus.CURRENT
    assert current.valid_to is None

    assert len(
        store.list(
            user_id="user_01",
            memory_type=MemoryType.SEMANTIC,
        )
    ) == 1


def test_equivalent_current_state_is_duplicate_noop():
    store, service = build_service()

    add_nottingham(service)

    result = service.ingest_state(
        user_id="user_01",
        state_key="user.residence",
        value="  NOTTINGHAM  ",
        text="The user still lives in Nottingham.",
        observed_at=utc(2026, 6, 2),
        source_turn_id="turn_02",
    )

    assert (
        result.operation
        == LifecycleOperation.NOOP_DUPLICATE
    )

    assert len(
        store.list(
            user_id="user_01",
            memory_type=MemoryType.SEMANTIC,
        )
    ) == 1


def test_newer_conflicting_state_supersedes_old_state():
    store, service = build_service()

    initial = add_nottingham(service)
    transition = move_to_london(service)

    assert (
        transition.operation
        == LifecycleOperation.SUPERSEDE_STATE
    )

    current = service.current_state(
        user_id="user_01",
        state_key="user.residence",
    )

    assert current is not None
    assert current.value == "London"
    assert current.status == MemoryStatus.CURRENT
    assert (
        current.supersedes
        == initial.current_memory_id
    )

    old = store.get(initial.current_memory_id)

    assert old is not None
    assert old.status == MemoryStatus.SUPERSEDED
    assert old.valid_to == utc(2026, 7, 15)
    assert (
        old.superseded_by
        == current.memory_id
    )


def test_state_transition_creates_episodic_memory():
    store, service = build_service()

    add_nottingham(service)
    transition = move_to_london(service)

    assert transition.episodic_memory_id is not None

    episode = store.get(
        transition.episodic_memory_id
    )

    assert episode is not None
    assert (
        episode.memory_type
        == MemoryType.EPISODIC
    )
    assert (
        episode.metadata["old_value"]
        == "Nottingham"
    )
    assert (
        episode.metadata["new_value"]
        == "London"
    )
    assert "moved" in episode.text.lower()


def test_current_query_exposes_only_current_state():
    _, service = build_service()

    add_nottingham(service)
    move_to_london(service)

    current = service.current_state(
        user_id="user_01",
        state_key="user.residence",
    )

    assert current is not None
    assert current.value == "London"
    assert current.status == MemoryStatus.CURRENT


def test_historical_query_preserves_old_state_and_event():
    _, service = build_service()

    add_nottingham(service)
    move_to_london(service)

    history = service.state_history(
        user_id="user_01",
        state_key="user.residence",
    )

    semantic_values = {
        record.value
        for record in history
        if (
            record.memory_type
            == MemoryType.SEMANTIC
        )
    }

    episodes = [
        record
        for record in history
        if (
            record.memory_type
            == MemoryType.EPISODIC
        )
    ]

    assert semantic_values == {
        "Nottingham",
        "London",
    }
    assert len(episodes) == 1
    assert "Nottingham" in episodes[0].text
    assert "London" in episodes[0].text


def test_older_contradictory_state_is_rejected():
    store, service = build_service()

    add_nottingham(service)

    result = service.ingest_state(
        user_id="user_01",
        state_key="user.residence",
        value="London",
        text="User lives in London.",
        observed_at=utc(2026, 5, 1),
        source_turn_id="older_turn",
    )

    assert (
        result.operation
        == LifecycleOperation.REJECT_OUT_OF_ORDER
    )

    current = service.current_state(
        user_id="user_01",
        state_key="user.residence",
    )

    assert current is not None
    assert current.value == "Nottingham"

    assert len(
        store.list(
            user_id="user_01",
            memory_type=MemoryType.SEMANTIC,
        )
    ) == 1


def test_exact_source_turn_replay_is_idempotent():
    store, service = build_service()

    first = add_nottingham(service)
    replay = add_nottingham(service)

    assert (
        replay.operation
        == LifecycleOperation.NOOP_REPLAY
    )
    assert (
        replay.current_memory_id
        == first.current_memory_id
    )

    assert len(store.all()) == 1


def test_source_turn_replay_with_new_value_fails():
    _, service = build_service()

    add_nottingham(service)

    with pytest.raises(
        LifecycleReplayConflict
    ):
        service.ingest_state(
            user_id="user_01",
            state_key="user.residence",
            value="London",
            text="User lives in London.",
            observed_at=utc(2026, 6, 1),
            source_turn_id="turn_01",
        )


def test_invariant_rejects_two_current_states():
    store = InMemoryLifecycleStore(
        [
            LifecycleRecord(
                memory_id="state_a",
                user_id="user_01",
                memory_type=MemoryType.SEMANTIC,
                text="User lives in Nottingham.",
                status=MemoryStatus.CURRENT,
                valid_from=utc(2026, 6, 1),
                state_key="user.residence",
                value="Nottingham",
            ),
            LifecycleRecord(
                memory_id="state_b",
                user_id="user_01",
                memory_type=MemoryType.SEMANTIC,
                text="User lives in London.",
                status=MemoryStatus.CURRENT,
                valid_from=utc(2026, 7, 1),
                state_key="user.residence",
                value="London",
            ),
        ]
    )

    service = LifecycleService(store)

    with pytest.raises(
        LifecycleInvariantError
    ):
        service.validate_invariants()
