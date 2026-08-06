#!/usr/bin/env python3
"""Real Neo4j integration smoke for C3 Phase II-B."""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.lifecycle import (
    LifecycleOperation,
    LifecycleService,
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


def serialise(value):
    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "value"):
        return value.value

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def main() -> None:
    required = [
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ]

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )

    run_id = (
        os.getenv("PHASE2B_RUN_ID")
        or uuid4().hex[:12]
    )

    user_id = (
        f"phase2b_neo4j_smoke_"
        f"{os.getenv('USER', 'user')}_{run_id}"
    )

    output_root = Path(
        os.getenv(
            "PHASE2B_OUTPUT_ROOT",
            (
                f"/data/{os.getenv('USER', 'user')}"
                "/c3_lite/evaluation/"
                "phase2b_neo4j_lifecycle_smoke_v01"
            ),
        )
    ) / run_id

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_path = output_root / "summary.json"
    database = (
        os.getenv("NEO4J_DATABASE")
        or None
    )

    store = None
    summary = {
        "run_id": run_id,
        "user_id": user_id,
        "database": database,
        "pipeline_passed": False,
        "cleanup_completed": False,
    }

    try:
        store = Neo4jLifecycleStore(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
            database=database,
        )

        store.driver.verify_connectivity()

        service = LifecycleService(store)

        print("========================================")
        print("PHASE II-B3 REAL NEO4J LIFECYCLE SMOKE")
        print("========================================")
        print("Run ID:", run_id)
        print("User ID:", user_id)
        print("Database:", database or "<default>")
        print("Neo4j connectivity: PASSED")
        print()

        initial = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="Nottingham",
            text="User lives in Nottingham.",
            observed_at=utc(2026, 6, 1),
            source_turn_id="turn_01",
            metadata={
                "subject": user_id,
                "predicate": "user.residence",
                "object": "Nottingham",
                "confidence": 1.0,
                "authority": "user_confirmed",
            },
        )

        print(
            "Initial operation:",
            initial.operation.value,
        )

        duplicate = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="Nottingham",
            text="User still lives in Nottingham.",
            observed_at=utc(2026, 6, 2),
            source_turn_id="turn_02",
            metadata={
                "subject": user_id,
                "predicate": "user.residence",
                "object": "Nottingham",
            },
        )

        print(
            "Duplicate operation:",
            duplicate.operation.value,
        )

        transition = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="London",
            text="User currently lives in London.",
            transition_text=(
                "User moved from Nottingham "
                "to London in July 2026."
            ),
            observed_at=utc(2026, 7, 15),
            source_turn_id="turn_03",
            metadata={
                "subject": user_id,
                "predicate": "user.residence",
                "object": "London",
                "confidence": 1.0,
                "authority": "user_confirmed",
            },
        )

        print(
            "Transition operation:",
            transition.operation.value,
        )

        service.validate_invariants()

        current = service.current_state(
            user_id=user_id,
            state_key="user.residence",
        )

        history = service.state_history(
            user_id=user_id,
            state_key="user.residence",
        )

        old_state = store.get(
            initial.current_memory_id
        )

        new_state = store.get(
            transition.current_memory_id
        )

        episode = store.get(
            transition.episodic_memory_id
        )

        with store._session() as session:
            relation_row = session.run(
                """
                MATCH
                  (new:SemanticFact)
                    -[sup:SUPERSEDES]->
                  (old:SemanticFact)
                WHERE new.fact_id = $new_id
                  AND old.fact_id = $old_id
                  AND new.lifecycle_managed = true
                  AND old.lifecycle_managed = true

                MATCH
                  (event:Episode)
                    -[from_rel:FROM_STATE]->
                  (old)

                MATCH
                  (event)
                    -[to_rel:TO_STATE]->
                  (new)

                WHERE event.episode_id = $episode_id
                  AND event.lifecycle_managed = true

                RETURN
                  count(DISTINCT sup)
                    AS supersedes_count,
                  count(DISTINCT from_rel)
                    AS from_state_count,
                  count(DISTINCT to_rel)
                    AS to_state_count
                """,
                old_id=initial.current_memory_id,
                new_id=transition.current_memory_id,
                episode_id=(
                    transition.episodic_memory_id
                ),
            ).single()

            count_row = session.run(
                """
                MATCH (node:SemanticFact)
                WHERE node.user_id = $user_id
                  AND node.lifecycle_managed = true
                  AND node.state_key = $state_key
                RETURN
                  count(node) AS semantic_count,
                  sum(
                    CASE
                      WHEN toLower(
                        coalesce(
                          node.status,
                          'current'
                        )
                      ) = 'current'
                      THEN 1
                      ELSE 0
                    END
                  ) AS current_count
                """,
                user_id=user_id,
                state_key="user.residence",
            ).single()

        relation_counts = {
            "supersedes": int(
                relation_row[
                    "supersedes_count"
                ]
            ),
            "from_state": int(
                relation_row[
                    "from_state_count"
                ]
            ),
            "to_state": int(
                relation_row[
                    "to_state_count"
                ]
            ),
        }

        semantic_count = int(
            count_row["semantic_count"]
        )

        current_count = int(
            count_row["current_count"]
        )

        checks = {
            "initial_add": (
                initial.operation
                == LifecycleOperation.ADD_INITIAL
            ),
            "duplicate_noop": (
                duplicate.operation
                == LifecycleOperation.NOOP_DUPLICATE
            ),
            "supersede_operation": (
                transition.operation
                == LifecycleOperation.SUPERSEDE_STATE
            ),
            "current_is_london": (
                current is not None
                and current.value == "London"
                and current.status
                == MemoryStatus.CURRENT
            ),
            "old_is_superseded": (
                old_state is not None
                and old_state.status
                == MemoryStatus.SUPERSEDED
                and old_state.superseded_by
                == transition.current_memory_id
            ),
            "new_supersedes_old": (
                new_state is not None
                and new_state.supersedes
                == initial.current_memory_id
            ),
            "episode_created": (
                episode is not None
                and episode.memory_type
                == MemoryType.EPISODIC
            ),
            "one_current_state": (
                current_count == 1
            ),
            "two_semantic_states": (
                semantic_count == 2
            ),
            "history_has_three_records": (
                len(history) == 3
            ),
            "supersedes_relation": (
                relation_counts["supersedes"] == 1
            ),
            "from_state_relation": (
                relation_counts["from_state"] == 1
            ),
            "to_state_relation": (
                relation_counts["to_state"] == 1
            ),
        }

        summary.update(
            {
                "pipeline_passed": all(
                    checks.values()
                ),
                "operations": {
                    "initial": (
                        initial.operation.value
                    ),
                    "duplicate": (
                        duplicate.operation.value
                    ),
                    "transition": (
                        transition.operation.value
                    ),
                },
                "memory_ids": {
                    "old_state": (
                        initial.current_memory_id
                    ),
                    "new_state": (
                        transition.current_memory_id
                    ),
                    "episode": (
                        transition.episodic_memory_id
                    ),
                },
                "current_value": (
                    current.value
                    if current is not None
                    else None
                ),
                "semantic_count": semantic_count,
                "current_count": current_count,
                "history_count": len(history),
                "relation_counts": relation_counts,
                "checks": checks,
            }
        )

        print()
        print("===== CHECKS =====")

        for name, passed in checks.items():
            print(f"{name}: {passed}")

        print()
        print(
            "Current state:",
            current.value if current else None,
        )
        print(
            "Semantic states:",
            semantic_count,
        )
        print(
            "History records:",
            len(history),
        )
        print(
            "Relations:",
            relation_counts,
        )

        if not summary["pipeline_passed"]:
            raise AssertionError(
                "One or more lifecycle checks failed."
            )

        print()
        print(
            "PHASE II-B3 REAL NEO4J "
            "LIFECYCLE SMOKE: PASSED"
        )

    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

        print(
            "PHASE II-B3 REAL NEO4J "
            "LIFECYCLE SMOKE: FAILED",
            file=sys.stderr,
        )

        raise

    finally:
        if store is not None:
            try:
                with store._session() as session:
                    cleanup_row = session.run(
                        """
                        MATCH (node)
                        WHERE node.user_id = $user_id
                          AND node.lifecycle_managed = true
                        WITH collect(node) AS nodes
                        FOREACH (
                          node IN nodes |
                          DETACH DELETE node
                        )
                        RETURN size(nodes)
                          AS deleted_count
                        """,
                        user_id=user_id,
                    ).single()

                summary["cleanup_deleted_count"] = int(
                    cleanup_row["deleted_count"]
                )

                summary["cleanup_completed"] = True

                print(
                    "Cleanup deleted nodes:",
                    summary[
                        "cleanup_deleted_count"
                    ],
                )

            finally:
                store.close()

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=serialise,
            ),
            encoding="utf-8",
        )

        print(
            "Summary:",
            summary_path,
        )


if __name__ == "__main__":
    main()
