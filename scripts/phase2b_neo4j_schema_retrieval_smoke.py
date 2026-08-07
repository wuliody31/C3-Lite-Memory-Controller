#!/usr/bin/env python3
"""Real Neo4j schema and lifecycle retrieval smoke."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from neo4j import GraphDatabase

from src.lifecycle import (
    LifecycleOperation,
    LifecycleService,
)
from src.lifecycle_neo4j import (
    Neo4jLifecycleStore,
)
from src.neo4j_schema import (
    Neo4jSchemaBootstrap,
)
from src.retrievers.neo4j_store import (
    Neo4jMemoryStore,
)
from src.schemas import (
    MemoryType,
    QueryFeatures,
    QueryMode,
    QueryState,
)


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


def query_features(
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
            "user lives residence"
        ),
        tokens=[
            "user",
            "lives",
            "residence",
        ],
        entities=["residence"],
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


def safe_schema_snapshot(
    driver,
    database: str | None,
) -> dict:
    kwargs = (
        {"database": database}
        if database
        else {}
    )

    with driver.session(**kwargs) as session:
        constraints = [
            dict(row)
            for row in session.run(
                """
                SHOW CONSTRAINTS
                YIELD name, type
                RETURN name, type
                ORDER BY name
                """
            )
        ]

        indexes = [
            dict(row)
            for row in session.run(
                """
                SHOW INDEXES
                YIELD name, type, state
                RETURN name, type, state
                ORDER BY name
                """
            )
        ]

    return {
        "constraints": constraints,
        "indexes": indexes,
    }


def duplicate_audit(
    driver,
    database: str | None,
) -> dict:
    kwargs = (
        {"database": database}
        if database
        else {}
    )

    with driver.session(**kwargs) as session:
        semantic = session.run(
            """
            MATCH (node:SemanticFact)
            WHERE node.fact_id IS NOT NULL
            WITH
                node.fact_id AS memory_id,
                count(*) AS occurrence_count
            WHERE occurrence_count > 1
            RETURN
                count(*) AS duplicate_groups,
                coalesce(
                    sum(occurrence_count),
                    0
                ) AS duplicate_nodes
            """
        ).single()

        episodic = session.run(
            """
            MATCH (node:Episode)
            WHERE node.episode_id IS NOT NULL
            WITH
                node.episode_id AS memory_id,
                count(*) AS occurrence_count
            WHERE occurrence_count > 1
            RETURN
                count(*) AS duplicate_groups,
                coalesce(
                    sum(occurrence_count),
                    0
                ) AS duplicate_nodes
            """
        ).single()

    return {
        "semantic": {
            "duplicate_groups": int(
                semantic["duplicate_groups"]
            ),
            "duplicate_nodes": int(
                semantic["duplicate_nodes"]
            ),
        },
        "episodic": {
            "duplicate_groups": int(
                episodic["duplicate_groups"]
            ),
            "duplicate_nodes": int(
                episodic["duplicate_nodes"]
            ),
        },
    }


def candidate_summary(candidate):
    return {
        "memory_id": candidate.memory_id,
        "status": candidate.status,
        "object_value": (
            candidate.object_value
        ),
        "relations": candidate.relations,
        "text": candidate.text,
    }


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
            "Missing environment variables: "
            + ", ".join(missing)
        )

    run_id = (
        os.getenv("PHASE2B_RUN_ID")
        or uuid4().hex[:12]
    )

    user_id = (
        f"phase2b_schema_retrieval_"
        f"{os.getenv('USER', 'user')}_"
        f"{run_id}"
    )

    database = (
        os.getenv("NEO4J_DATABASE")
        or None
    )

    output_root = Path(
        os.getenv(
            "PHASE2B_OUTPUT_ROOT",
            (
                f"/data/{os.getenv('USER', 'user')}"
                "/c3_lite/evaluation/"
                "phase2b_neo4j_schema_retrieval_v01"
            ),
        )
    ) / run_id

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary_path = (
        output_root / "summary.json"
    )

    summary = {
        "run_id": run_id,
        "user_id": user_id,
        "database": database,
        "pipeline_passed": False,
        "cleanup_completed": False,
    }

    lifecycle_store = None
    driver = None

    try:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(
                os.environ["NEO4J_USER"],
                os.environ[
                    "NEO4J_PASSWORD"
                ],
            ),
        )

        driver.verify_connectivity()

        print(
            "========================================"
        )
        print(
            "PHASE II-B4 REAL NEO4J "
            "SCHEMA + RETRIEVAL SMOKE"
        )
        print(
            "========================================"
        )
        print("Run ID:", run_id)
        print("User ID:", user_id)
        print(
            "Database:",
            database or "<default>",
        )
        print("Neo4j connectivity: PASSED")

        duplicate_result = duplicate_audit(
            driver,
            database,
        )

        summary[
            "duplicate_audit"
        ] = duplicate_result

        print()
        print("===== DUPLICATE AUDIT =====")
        print(
            json.dumps(
                duplicate_result,
                indent=2,
            )
        )

        duplicate_total = sum(
            section["duplicate_groups"]
            for section
            in duplicate_result.values()
        )

        if duplicate_total != 0:
            raise RuntimeError(
                "Duplicate fact_id or episode_id "
                "values detected. Schema mutation "
                "was aborted."
            )

        summary["schema_before"] = (
            safe_schema_snapshot(
                driver,
                database,
            )
        )

        bootstrap = Neo4jSchemaBootstrap(
            driver=driver,
            database=database,
        )

        bootstrap_result = bootstrap.apply(
            wait_seconds=60
        )

        schema_inspection = (
            bootstrap.inspect()
        )

        summary[
            "bootstrap_result"
        ] = bootstrap_result

        summary[
            "schema_inspection"
        ] = schema_inspection

        print()
        print("===== INSTALLED SCHEMA =====")
        print(
            json.dumps(
                schema_inspection,
                indent=2,
            )
        )

        expected_constraints = {
            "c3_semantic_fact_id_unique",
            "c3_episode_id_unique",
        }

        expected_indexes = {
            "c3_semantic_lifecycle_lookup",
            "c3_episode_lifecycle_lookup",
            "semantic_fact_text",
            "episode_text",
        }

        installed_constraints = {
            row["name"]
            for row
            in schema_inspection[
                "constraints"
            ]
        }

        installed_indexes = {
            row["name"]
            for row
            in schema_inspection[
                "indexes"
            ]
        }

        online_indexes = {
            row["name"]
            for row
            in schema_inspection[
                "indexes"
            ]
            if str(
                row.get("state", "")
            ).upper() == "ONLINE"
        }

        schema_checks = {
            "semantic_constraint_present": (
                "c3_semantic_fact_id_unique"
                in installed_constraints
            ),
            "episode_constraint_present": (
                "c3_episode_id_unique"
                in installed_constraints
            ),
            "all_expected_indexes_present": (
                expected_indexes
                <= installed_indexes
            ),
            "all_expected_indexes_online": (
                expected_indexes
                <= online_indexes
            ),
        }

        summary[
            "schema_checks"
        ] = schema_checks

        if not all(
            schema_checks.values()
        ):
            raise AssertionError(
                "Schema bootstrap checks failed."
            )

        lifecycle_store = (
            Neo4jLifecycleStore(
                uri=os.environ[
                    "NEO4J_URI"
                ],
                user=os.environ[
                    "NEO4J_USER"
                ],
                password=os.environ[
                    "NEO4J_PASSWORD"
                ],
                database=database,
            )
        )

        service = LifecycleService(
            lifecycle_store
        )

        initial = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="Nottingham",
            text=(
                "User lives in Nottingham."
            ),
            observed_at=utc(
                2026,
                6,
                1,
            ),
            source_turn_id="turn_01",
            metadata={
                "subject": user_id,
                "predicate": (
                    "user.residence"
                ),
                "object": "Nottingham",
                "authority": (
                    "user_confirmed"
                ),
                "confidence": 1.0,
            },
        )

        duplicate = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="Nottingham",
            text=(
                "User lives in Nottingham."
            ),
            observed_at=utc(
                2026,
                6,
                2,
            ),
            source_turn_id="turn_02",
            metadata={
                "subject": user_id,
                "predicate": (
                    "user.residence"
                ),
                "object": "Nottingham",
            },
        )

        transition = service.ingest_state(
            user_id=user_id,
            state_key="user.residence",
            value="London",
            text=(
                "User lives in London."
            ),
            transition_text=(
                "User moved from Nottingham "
                "to London in July 2026."
            ),
            observed_at=utc(
                2026,
                7,
                15,
            ),
            source_turn_id="turn_03",
            metadata={
                "subject": user_id,
                "predicate": (
                    "user.residence"
                ),
                "object": "London",
                "authority": (
                    "user_confirmed"
                ),
                "confidence": 1.0,
            },
        )

        retriever = Neo4jMemoryStore(
            config={
                "neo4j": {
                    "database": database,
                    "semantic_fulltext_index": (
                        "semantic_fact_text"
                    ),
                    "episode_fulltext_index": (
                        "episode_text"
                    ),
                }
            },
            database=database,
            driver=driver,
        )

        current_result = []
        historical_result = []

        for attempt in range(1, 7):
            current_result = (
                retriever.retrieve(
                    memory_type=(
                        MemoryType.SEMANTIC
                    ),
                    state=QueryState(
                        query=(
                            "Where does the "
                            "user live?"
                        ),
                        user_id=user_id,
                        current_time=utc(
                            2026,
                            8,
                            6,
                        ),
                    ),
                    features=query_features(
                        QueryMode.CURRENT
                    ),
                    top_k=10,
                )
            )

            historical_result = (
                retriever.retrieve(
                    memory_type=(
                        MemoryType.SEMANTIC
                    ),
                    state=QueryState(
                        query=(
                            "Where did the user "
                            "live before?"
                        ),
                        user_id=user_id,
                        current_time=utc(
                            2026,
                            8,
                            6,
                        ),
                    ),
                    features=query_features(
                        QueryMode.HISTORICAL
                    ),
                    top_k=10,
                )
            )

            current_ids = {
                item.memory_id
                for item in current_result
            }

            historical_ids = {
                item.memory_id
                for item
                in historical_result
            }

            if (
                transition.current_memory_id
                in current_ids
                and initial.current_memory_id
                in historical_ids
                and transition.current_memory_id
                in historical_ids
            ):
                break

            time.sleep(0.5)

        current_ids = {
            item.memory_id
            for item in current_result
        }

        historical_ids = {
            item.memory_id
            for item in historical_result
        }

        new_candidate = next(
            (
                item
                for item in current_result
                if item.memory_id
                == transition.current_memory_id
            ),
            None,
        )

        relation_preserved = bool(
            new_candidate
            and any(
                relation.get("type")
                == "SUPERSEDES"
                and relation.get(
                    "target_id"
                )
                == initial.current_memory_id
                for relation
                in new_candidate.relations
            )
        )

        retrieval_checks = {
            "initial_operation": (
                initial.operation
                == LifecycleOperation.ADD_INITIAL
            ),
            "duplicate_operation": (
                duplicate.operation
                == LifecycleOperation.NOOP_DUPLICATE
            ),
            "transition_operation": (
                transition.operation
                == LifecycleOperation.SUPERSEDE_STATE
            ),
            "current_contains_london": (
                transition.current_memory_id
                in current_ids
            ),
            "current_excludes_nottingham": (
                initial.current_memory_id
                not in current_ids
            ),
            "historical_contains_london": (
                transition.current_memory_id
                in historical_ids
            ),
            "historical_contains_nottingham": (
                initial.current_memory_id
                in historical_ids
            ),
            "supersedes_relation_preserved": (
                relation_preserved
            ),
        }

        summary.update(
            {
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
                    "nottingham": (
                        initial.current_memory_id
                    ),
                    "london": (
                        transition.current_memory_id
                    ),
                    "episode": (
                        transition.episodic_memory_id
                    ),
                },
                "current_results": [
                    candidate_summary(item)
                    for item in current_result
                ],
                "historical_results": [
                    candidate_summary(item)
                    for item
                    in historical_result
                ],
                "retrieval_checks": (
                    retrieval_checks
                ),
            }
        )

        all_checks = {
            **schema_checks,
            **retrieval_checks,
        }

        summary["checks"] = all_checks
        summary["pipeline_passed"] = all(
            all_checks.values()
        )

        print()
        print("===== CHECKS =====")

        for name, passed in (
            all_checks.items()
        ):
            print(f"{name}: {passed}")

        print()
        print(
            "Current result IDs:",
            sorted(current_ids),
        )
        print(
            "Historical result IDs:",
            sorted(historical_ids),
        )

        if not summary[
            "pipeline_passed"
        ]:
            raise AssertionError(
                "One or more schema or "
                "retrieval checks failed."
            )

        print()
        print(
            "PHASE II-B4 REAL NEO4J "
            "SCHEMA + RETRIEVAL SMOKE: "
            "PASSED"
        )

    except Exception as exc:
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": (
                traceback.format_exc()
            ),
        }

        print()
        print(
            "PHASE II-B4 REAL NEO4J "
            "SCHEMA + RETRIEVAL SMOKE: "
            "FAILED",
            file=sys.stderr,
        )

        raise

    finally:
        if driver is not None:
            try:
                kwargs = (
                    {"database": database}
                    if database
                    else {}
                )

                with driver.session(
                    **kwargs
                ) as session:
                    row = session.run(
                        """
                        MATCH (node)
                        WHERE
                            node.user_id
                            = $user_id
                          AND
                            node.lifecycle_managed
                            = true
                        WITH collect(node)
                            AS nodes
                        FOREACH (
                            node IN nodes |
                            DETACH DELETE node
                        )
                        RETURN size(nodes)
                            AS deleted_count
                        """,
                        user_id=user_id,
                    ).single()

                summary[
                    "cleanup_deleted_count"
                ] = int(
                    row["deleted_count"]
                )

                summary[
                    "cleanup_completed"
                ] = True

                print(
                    "Cleanup deleted nodes:",
                    summary[
                        "cleanup_deleted_count"
                    ],
                )

            except Exception as cleanup_exc:
                summary[
                    "cleanup_error"
                ] = {
                    "type": type(
                        cleanup_exc
                    ).__name__,
                    "message": str(
                        cleanup_exc
                    ),
                }

        if lifecycle_store is not None:
            lifecycle_store.close()

        elif driver is not None:
            driver.close()

        summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(
            "Summary:",
            summary_path,
        )


if __name__ == "__main__":
    main()
