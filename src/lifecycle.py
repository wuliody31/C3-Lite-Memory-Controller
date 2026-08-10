"""Backend-neutral memory lifecycle kernel for C3 Phase II-B.

This module deliberately contains no LLM, Neo4j, Qdrant or GPU
dependency. It defines deterministic state-transition semantics that
can later be executed through different persistence adapters.
"""

from __future__ import annotations

import builtins

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Protocol
from uuid import NAMESPACE_URL, uuid5

from .schemas import MemoryType


class MemoryStatus(str, Enum):
    """Lifecycle status compatible with the existing C3 retrievers."""

    CURRENT = "current"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class LifecycleOperation(str, Enum):
    """Result of ingesting one state-bearing observation."""

    ADD_INITIAL = "add_initial"
    NOOP_DUPLICATE = "noop_duplicate"
    NOOP_REPLAY = "noop_replay"
    SUPERSEDE_STATE = "supersede_state"
    REJECT_OUT_OF_ORDER = "reject_out_of_order"


class LifecycleError(RuntimeError):
    """Base class for lifecycle failures."""


class LifecycleInvariantError(LifecycleError):
    """Raised when persisted lifecycle records violate invariants."""


class LifecycleReplayConflict(LifecycleError):
    """Raised when one source turn is replayed with different content."""


class LifecycleStoreError(LifecycleError):
    """Raised for invalid backend mutations."""


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """Canonical lifecycle record used by the control plane."""

    memory_id: str
    user_id: str
    memory_type: MemoryType
    text: str
    status: MemoryStatus
    valid_from: datetime
    state_key: str | None = None
    value: str | None = None
    valid_to: datetime | None = None
    source_turn_id: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateTransition:
    """Auditable decision returned by the lifecycle controller."""

    operation: LifecycleOperation
    user_id: str
    state_key: str
    previous_memory_id: str | None
    current_memory_id: str | None
    episodic_memory_id: str | None
    reason: str


class LifecycleStore(Protocol):
    """Persistence contract required by the lifecycle controller."""

    def get(self, memory_id: str) -> LifecycleRecord | None:
        ...

    def list(
        self,
        *,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        state_key: str | None = None,
        status: MemoryStatus | None = None,
    ) -> list[LifecycleRecord]:
        ...

    def all(self) -> builtins.list[LifecycleRecord]:
        ...

    def commit(
        self,
        *,
        add: Iterable[LifecycleRecord] = (),
        update: Iterable[LifecycleRecord] = (),
    ) -> None:
        """Apply one lifecycle transition atomically."""
        ...


class InMemoryLifecycleStore:
    """Deterministic atomic store for CPU tests and local simulation."""

    def __init__(
        self,
        records: Iterable[LifecycleRecord] = (),
    ) -> None:
        self._records: dict[str, LifecycleRecord] = {}

        for record in records:
            if record.memory_id in self._records:
                raise LifecycleStoreError(
                    f"Duplicate memory ID: {record.memory_id}"
                )
            self._records[record.memory_id] = record

    def get(
        self,
        memory_id: str,
    ) -> LifecycleRecord | None:
        return self._records.get(memory_id)

    def list(
        self,
        *,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        state_key: str | None = None,
        status: MemoryStatus | None = None,
    ) -> list[LifecycleRecord]:
        output = []

        for record in self._records.values():
            if user_id is not None and record.user_id != user_id:
                continue

            if (
                memory_type is not None
                and record.memory_type != memory_type
            ):
                continue

            if (
                state_key is not None
                and record.state_key != state_key
            ):
                continue

            if status is not None and record.status != status:
                continue

            output.append(record)

        return sorted(
            output,
            key=lambda item: (
                item.valid_from,
                item.memory_id,
            ),
        )

    def all(self) -> builtins.list[LifecycleRecord]:
        return self.list()

    def commit(
        self,
        *,
        add: Iterable[LifecycleRecord] = (),
        update: Iterable[LifecycleRecord] = (),
    ) -> None:
        """Use copy-on-write so a failed mutation changes nothing."""

        staged = dict(self._records)
        updates = list(update)
        additions = list(add)

        for record in updates:
            if record.memory_id not in staged:
                raise LifecycleStoreError(
                    "Cannot update missing memory: "
                    f"{record.memory_id}"
                )

            staged[record.memory_id] = record

        for record in additions:
            if record.memory_id in staged:
                raise LifecycleStoreError(
                    "Cannot add duplicate memory: "
                    f"{record.memory_id}"
                )

            staged[record.memory_id] = record

        self._records = staged


def _normalise_value(value: str) -> str:
    return " ".join(value.split()).casefold()


def _require_nonempty(
    value: str,
    field_name: str,
) -> str:
    clean = value.strip()

    if not clean:
        raise ValueError(
            f"{field_name} must be non-empty."
        )

    return clean


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "observed_at must be timezone-aware."
        )

    return value.astimezone(timezone.utc)


def _stable_id(
    *parts: object,
) -> str:
    canonical = "|".join(str(part) for part in parts)
    return str(
        uuid5(
            NAMESPACE_URL,
            f"c3-lifecycle:{canonical}",
        )
    )


class LifecycleService:
    """Automatic semantic-state transition controller."""

    def __init__(
        self,
        store: LifecycleStore,
    ) -> None:
        self.store = store

    def ingest_state(
        self,
        *,
        user_id: str,
        state_key: str,
        value: str,
        text: str,
        observed_at: datetime,
        source_turn_id: str,
        transition_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Ingest one state-bearing observation.

        Rules:
        1. One user/state_key has at most one CURRENT semantic record.
        2. Exact source-turn replay is idempotent.
        3. Same normalised value is a duplicate no-op.
        4. An older contradictory observation is rejected.
        5. A newer contradictory value supersedes the current semantic
           state and creates an episodic transition record.
        """

        user_id = _require_nonempty(
            user_id,
            "user_id",
        )

        state_key = _require_nonempty(
            state_key,
            "state_key",
        )

        value = _require_nonempty(
            value,
            "value",
        )

        text = _require_nonempty(
            text,
            "text",
        )

        source_turn_id = _require_nonempty(
            source_turn_id,
            "source_turn_id",
        )

        observed_at = _as_utc(observed_at)
        normalised_value = _normalise_value(value)
        metadata = dict(metadata or {})

        existing_states = self.store.list(
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            state_key=state_key,
        )

        replay_records = [
            record
            for record in existing_states
            if record.source_turn_id == source_turn_id
        ]

        if replay_records:
            replay = replay_records[0]

            if (
                replay.value is None
                or _normalise_value(replay.value)
                != normalised_value
            ):
                raise LifecycleReplayConflict(
                    "The same source_turn_id was replayed "
                    "with a different state value: "
                    f"{source_turn_id}"
                )

            current = self.current_state(
                user_id=user_id,
                state_key=state_key,
            )

            return StateTransition(
                operation=LifecycleOperation.NOOP_REPLAY,
                user_id=user_id,
                state_key=state_key,
                previous_memory_id=replay.supersedes,
                current_memory_id=(
                    current.memory_id
                    if current is not None
                    else replay.memory_id
                ),
                episodic_memory_id=None,
                reason=(
                    "source_turn_id has already been "
                    "committed"
                ),
            )

        current = self.current_state(
            user_id=user_id,
            state_key=state_key,
        )

        if current is None:
            new_id = _stable_id(
                "semantic",
                user_id,
                state_key,
                source_turn_id,
                normalised_value,
                observed_at.isoformat(),
            )

            new_record = LifecycleRecord(
                memory_id=new_id,
                user_id=user_id,
                memory_type=MemoryType.SEMANTIC,
                text=text,
                status=MemoryStatus.CURRENT,
                valid_from=observed_at,
                state_key=state_key,
                value=value,
                source_turn_id=source_turn_id,
                metadata={
                    **metadata,
                    "lifecycle_role": "semantic_state",
                    "operation": (
                        LifecycleOperation.ADD_INITIAL.value
                    ),
                },
            )

            self.store.commit(add=[new_record])
            self.validate_invariants()

            return StateTransition(
                operation=LifecycleOperation.ADD_INITIAL,
                user_id=user_id,
                state_key=state_key,
                previous_memory_id=None,
                current_memory_id=new_id,
                episodic_memory_id=None,
                reason="no current state existed",
            )

        if (
            current.value is not None
            and _normalise_value(current.value)
            == normalised_value
        ):
            return StateTransition(
                operation=LifecycleOperation.NOOP_DUPLICATE,
                user_id=user_id,
                state_key=state_key,
                previous_memory_id=current.memory_id,
                current_memory_id=current.memory_id,
                episodic_memory_id=None,
                reason=(
                    "new observation is semantically "
                    "equivalent to current state"
                ),
            )

        if observed_at <= current.valid_from:
            return StateTransition(
                operation=(
                    LifecycleOperation.REJECT_OUT_OF_ORDER
                ),
                user_id=user_id,
                state_key=state_key,
                previous_memory_id=current.memory_id,
                current_memory_id=current.memory_id,
                episodic_memory_id=None,
                reason=(
                    "contradictory observation is not "
                    "newer than current state"
                ),
            )

        new_id = _stable_id(
            "semantic",
            user_id,
            state_key,
            source_turn_id,
            normalised_value,
            observed_at.isoformat(),
        )

        episode_id = _stable_id(
            "episodic-transition",
            user_id,
            state_key,
            current.memory_id,
            new_id,
            observed_at.isoformat(),
        )

        superseded_record = replace(
            current,
            status=MemoryStatus.SUPERSEDED,
            valid_to=observed_at,
            superseded_by=new_id,
            metadata={
                **current.metadata,
                "lifecycle_role": "semantic_state",
                "superseded_by": new_id,
                "superseded_at": (
                    observed_at.isoformat()
                ),
            },
        )

        new_record = LifecycleRecord(
            memory_id=new_id,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            text=text,
            status=MemoryStatus.CURRENT,
            valid_from=observed_at,
            state_key=state_key,
            value=value,
            source_turn_id=source_turn_id,
            supersedes=current.memory_id,
            metadata={
                **metadata,
                "lifecycle_role": "semantic_state",
                "operation": (
                    LifecycleOperation
                    .SUPERSEDE_STATE
                    .value
                ),
                "supersedes": current.memory_id,
            },
        )

        episode_text = (
            transition_text.strip()
            if transition_text
            else (
                f"{state_key} changed from "
                f"{current.value} to {value}."
            )
        )

        episodic_record = LifecycleRecord(
            memory_id=episode_id,
            user_id=user_id,
            memory_type=MemoryType.EPISODIC,
            text=episode_text,
            status=MemoryStatus.CURRENT,
            valid_from=observed_at,
            state_key=state_key,
            value=None,
            source_turn_id=source_turn_id,
            metadata={
                **metadata,
                "lifecycle_role": "state_transition",
                "state_key": state_key,
                "old_value": current.value,
                "new_value": value,
                "old_memory_id": current.memory_id,
                "new_memory_id": new_id,
            },
        )

        self.store.commit(
            update=[superseded_record],
            add=[
                new_record,
                episodic_record,
            ],
        )

        self.validate_invariants()

        return StateTransition(
            operation=LifecycleOperation.SUPERSEDE_STATE,
            user_id=user_id,
            state_key=state_key,
            previous_memory_id=current.memory_id,
            current_memory_id=new_id,
            episodic_memory_id=episode_id,
            reason=(
                "newer contradictory state superseded "
                "the current state"
            ),
        )

    def current_state(
        self,
        *,
        user_id: str,
        state_key: str,
    ) -> LifecycleRecord | None:
        records = self.store.list(
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            state_key=state_key,
            status=MemoryStatus.CURRENT,
        )

        if len(records) > 1:
            raise LifecycleInvariantError(
                "More than one current semantic state "
                f"for user={user_id!r}, "
                f"state_key={state_key!r}."
            )

        return records[0] if records else None

    def state_history(
        self,
        *,
        user_id: str,
        state_key: str,
    ) -> list[LifecycleRecord]:
        semantic = self.store.list(
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            state_key=state_key,
        )

        episodic = [
            record
            for record in self.store.list(
                user_id=user_id,
                memory_type=MemoryType.EPISODIC,
                state_key=state_key,
            )
            if (
                record.metadata.get("lifecycle_role")
                == "state_transition"
            )
        ]

        return sorted(
            [*semantic, *episodic],
            key=lambda record: (
                record.valid_from,
                0
                if record.memory_type
                == MemoryType.SEMANTIC
                else 1,
                record.memory_id,
            ),
        )

    def validate_invariants(self) -> None:
        records = self.store.all()
        by_id = {
            record.memory_id: record
            for record in records
        }

        groups: dict[
            tuple[str, str],
            list[LifecycleRecord],
        ] = {}

        for record in records:
            if (
                record.memory_type
                != MemoryType.SEMANTIC
                or record.state_key is None
            ):
                continue

            groups.setdefault(
                (
                    record.user_id,
                    record.state_key,
                ),
                [],
            ).append(record)

            if record.status == MemoryStatus.CURRENT:
                if record.valid_to is not None:
                    raise LifecycleInvariantError(
                        "Current state cannot have valid_to: "
                        f"{record.memory_id}"
                    )

            if (
                record.status
                == MemoryStatus.SUPERSEDED
            ):
                if record.valid_to is None:
                    raise LifecycleInvariantError(
                        "Superseded state requires valid_to: "
                        f"{record.memory_id}"
                    )

                if record.superseded_by is None:
                    raise LifecycleInvariantError(
                        "Superseded state requires "
                        "superseded_by: "
                        f"{record.memory_id}"
                    )

            if record.superseded_by is not None:
                target = by_id.get(
                    record.superseded_by
                )

                if target is None:
                    raise LifecycleInvariantError(
                        "superseded_by target is missing: "
                        f"{record.memory_id}"
                    )

                if target.supersedes != record.memory_id:
                    raise LifecycleInvariantError(
                        "SUPERSEDES relation is not "
                        "bidirectionally consistent."
                    )

            if record.supersedes is not None:
                source = by_id.get(record.supersedes)

                if source is None:
                    raise LifecycleInvariantError(
                        "supersedes source is missing: "
                        f"{record.memory_id}"
                    )

                if (
                    source.superseded_by
                    != record.memory_id
                ):
                    raise LifecycleInvariantError(
                        "SUPERSEDES relation is not "
                        "bidirectionally consistent."
                    )

        for (
            user_id,
            state_key,
        ), group in groups.items():
            current_count = sum(
                record.status
                == MemoryStatus.CURRENT
                for record in group
            )

            if current_count > 1:
                raise LifecycleInvariantError(
                    "At most one current state is "
                    "allowed for "
                    f"user={user_id!r}, "
                    f"state_key={state_key!r}."
                )
