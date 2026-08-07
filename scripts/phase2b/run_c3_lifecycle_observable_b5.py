#!/usr/bin/env python3
"""C3 lifecycle observable-retrieval runner for Phase II-B5.

Observable comparison stage:
    direct Neo4jMemoryStore semantic retrieval.

The comparison intentionally measures lifecycle-aware
semantic retrieval before downstream conflict resolution,
evidence selection, or answer generation.

The runner writes the same current_search_texts /
historical_search_texts interface consumed by the frozen
Mem0 lifecycle evaluator v0.2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from neo4j import GraphDatabase

from src.config import load_config
from src.lifecycle import LifecycleService
from src.lifecycle_neo4j import Neo4jLifecycleStore
from src.neo4j_schema import Neo4jSchemaBootstrap
from src.retrievers.neo4j_store import Neo4jMemoryStore
from src.schemas import (
    MemoryType,
    QueryFeatures,
    QueryMode,
    QueryState,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=Path,
        default=(
            ROOT
            / "data"
            / "lifecycle_benchmark_v01"
            / "cases.jsonl"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "c3_lite_v2_2_final.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--surface-variant",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--state-key",
        default=None,
    )

    parser.add_argument(
        "--pattern",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def parse_time(value: Any) -> datetime:
    text = str(value).strip()

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    result = datetime.fromisoformat(text)

    if result.tzinfo is None:
        raise ValueError(
            f"Timestamp is not timezone-aware: {value}"
        )

    return result.astimezone(timezone.utc)


def slug(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        value,
    ).strip("_")


def select_cases(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected = list(rows)

    if args.surface_variant is not None:
        selected = [
            row
            for row in selected
            if int(row["surface_variant"])
            == args.surface_variant
        ]

    if args.state_key is not None:
        selected = [
            row
            for row in selected
            if row["state_key"]
            == args.state_key
        ]

    if args.pattern is not None:
        selected = [
            row
            for row in selected
            if row["pattern"]
            == args.pattern
        ]

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        raise RuntimeError(
            "Case selection produced zero cases."
        )

    return selected


def event_source_id(
    case_id: str,
    index: int,
    event: dict[str, Any],
) -> str:
    for key in (
        "source_turn_id",
        "turn_id",
        "event_id",
    ):
        value = event.get(key)

        if value:
            return str(value)

    return f"{case_id}:event:{index}"


def cleanup_user(
    driver,
    database: str | None,
    user_id: str,
) -> int:
    kwargs = (
        {"database": database}
        if database
        else {}
    )

    with driver.session(**kwargs) as session:
        row = session.run(
            """
            MATCH (node)
            WHERE node.lifecycle_managed = true
              AND node.user_id = $user_id
            WITH collect(node) AS nodes
            WITH nodes, size(nodes) AS deleted
            FOREACH (
                node IN nodes |
                DETACH DELETE node
            )
            RETURN deleted
            """,
            user_id=user_id,
        ).single()

    return int(row["deleted"] or 0)


def record_texts(
    lifecycle_store: Neo4jLifecycleStore,
    memory_ids: list[str],
) -> list[str]:
    output: list[str] = []

    for memory_id in memory_ids:
        record = lifecycle_store.get(
            memory_id
        )

        if record is None:
            raise RuntimeError(
                "Ranked candidate is not present in "
                "the lifecycle store: "
                f"{memory_id}"
            )

        output.append(record.text)

    return output


def checkpoint(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    total_cases: int,
) -> None:
    results_path = (
        output_dir / "results.jsonl"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    last = rows[-1]

    progress = {
        "completed_cases": len(rows),
        "total_cases": total_cases,
        "last_case_id": last["case_id"],
        "last_pattern": last["pattern"],
        "last_state_key": (
            last["state_key"]
        ),
    }

    (
        output_dir / "progress.json"
    ).write_text(
        json.dumps(
            progress,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_case(
    *,
    case: dict[str, Any],
    run_id: str,
    service: LifecycleService,
    lifecycle_store: Neo4jLifecycleStore,
    retriever: Neo4jMemoryStore,
    driver,
    database: str | None,
) -> dict[str, Any]:
    case_id = str(case["case_id"])

    user_id = (
        "phase2b_c3_observable_"
        + slug(run_id)
        + "_"
        + slug(case_id)
    )

    # Defensive cleanup makes rerunning a smoke with the same
    # run ID deterministic.
    cleanup_user(
        driver,
        database,
        user_id,
    )

    events = list(case["events"])

    if not events:
        raise RuntimeError(
            f"Case has no events: {case_id}"
        )

    observed_times: list[datetime] = []
    operations: list[str] = []

    try:
        for index, event in enumerate(
            events,
            start=1,
        ):
            observed_at = parse_time(
                event["observed_at"]
            )

            observed_times.append(
                observed_at
            )

            transition = service.ingest_state(
                user_id=user_id,
                state_key=str(
                    case["state_key"]
                ),
                value=str(
                    event["value"]
                ),
                text=str(
                    event["text"]
                ),
                observed_at=observed_at,
                source_turn_id=event_source_id(
                    case_id,
                    index,
                    event,
                ),
                metadata={
                    "benchmark":
                        "lifecycle_benchmark_v01",
                    "case_id": case_id,
                    "pattern": case["pattern"],
                    "surface_variant":
                        case["surface_variant"],
                    "event_index": index,
                },
            )

            operations.append(
                transition.operation.value
            )

        expected_operations = list(
            case["expected_operations"]
        )

        if operations != expected_operations:
            raise RuntimeError(
                "Lifecycle operation mismatch for "
                f"{case_id}: "
                f"actual={operations}, "
                f"expected={expected_operations}"
            )

        # Query after the latest observation time rather than
        # using wall-clock time. This keeps the benchmark
        # deterministic.
        query_time = (
            max(observed_times)
            + timedelta(seconds=1)
        )

        current_state = QueryState(
            query=str(
                case["current_query"]
            ),
            user_id=user_id,
            current_time=query_time,
        )

        current_features = QueryFeatures(
            normalised_query=(
                current_state.query
                .strip()
                .lower()
            ),
            tokens=re.findall(
                r"[A-Za-z0-9_]+",
                current_state.query.lower(),
            ),
            entities=[],
            temporal_expressions=[],
            query_mode=QueryMode.CURRENT,
            task_type=None,
            asks_current_state=True,
            asks_historical_state=False,
            asks_timeline=False,
            asks_procedure=False,
            asks_explanation=False,
            asks_conflict=False,
            information_needs=[
                "current valid state"
            ],
        )

        current_candidates = retriever.retrieve(
            memory_type=MemoryType.SEMANTIC,
            state=current_state,
            features=current_features,
            top_k=20,
            include_archived=False,
        )

        historical_state = QueryState(
            query=str(
                case["historical_query"]
            ),
            user_id=user_id,
            current_time=query_time,
        )

        historical_features = QueryFeatures(
            normalised_query=(
                historical_state.query
                .strip()
                .lower()
            ),
            tokens=re.findall(
                r"[A-Za-z0-9_]+",
                historical_state.query.lower(),
            ),
            entities=[],
            temporal_expressions=[],
            query_mode=QueryMode.HISTORICAL,
            task_type=None,
            asks_current_state=False,
            asks_historical_state=True,
            asks_timeline=False,
            asks_procedure=False,
            asks_explanation=False,
            asks_conflict=False,
            information_needs=[
                "historical event or state"
            ],
        )

        historical_candidates = retriever.retrieve(
            memory_type=MemoryType.SEMANTIC,
            state=historical_state,
            features=historical_features,
            top_k=20,
            include_archived=False,
        )

        current_search_texts = [
            item.text
            for item in current_candidates
        ]

        historical_search_texts = [
            item.text
            for item in historical_candidates
        ]

        semantic_records = (
            lifecycle_store.list(
                user_id=user_id,
                memory_type=(
                    MemoryType.SEMANTIC
                ),
            )
        )

        episodic_records = (
            lifecycle_store.list(
                user_id=user_id,
                memory_type=(
                    MemoryType.EPISODIC
                ),
            )
        )

        # The frozen evaluator's final_memory_texts is
        # state-memory scope. Transition episodes are
        # recorded separately below rather than inflating
        # the semantic memory-growth metric.
        final_memory_texts = [
            record.text
            for record in semantic_records
        ]

        row = {
            "case_id": case_id,
            "pattern": case["pattern"],
            "state_key": case["state_key"],
            "surface_variant": (
                case["surface_variant"]
            ),

            "system":
                "C3 lifecycle + Neo4j",

            "observable_stage":
                "neo4j_semantic_retrieval_top20_role_conditioned",

            "user_id": user_id,

            "current_query":
                case["current_query"],

            "historical_query":
                case["historical_query"],

            "query_time":
                query_time.isoformat(),

            "operations": operations,

            "current_search_texts":
                current_search_texts,

            "historical_search_texts":
                historical_search_texts,

            "final_memory_texts":
                final_memory_texts,

            "current_retrieved_ids": [
                item.memory_id
                for item in current_candidates
            ],

            "current_retrieved_statuses": [
                item.status
                for item in current_candidates
            ],

            "current_retrieved_values": [
                item.object_value
                for item in current_candidates
            ],

            "historical_retrieved_ids": [
                item.memory_id
                for item in historical_candidates
            ],

            "historical_retrieved_statuses": [
                item.status
                for item in historical_candidates
            ],

            "historical_retrieved_values": [
                item.object_value
                for item in historical_candidates
            ],

            "final_semantic_count":
                len(semantic_records),

            "final_episodic_count":
                len(episodic_records),

            "final_all_lifecycle_count":
                (
                    len(semantic_records)
                    + len(episodic_records)
                ),

            "final_episodic_texts": [
                record.text
                for record in episodic_records
            ],
        }

        return row

    finally:
        cleanup_user(
            driver,
            database,
            user_id,
        )


def main() -> None:
    args = parse_args()

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
            "Missing Neo4j environment variables: "
            + ", ".join(missing)
        )

    cases = select_cases(
        read_jsonl(args.cases),
        args,
    )

    config = load_config(
        str(args.config)
    )

    database = (
        os.getenv("NEO4J_DATABASE")
        or config.get(
            "neo4j",
            {},
        ).get("database")
        or None
    )

    run_id = (
        os.getenv("C3_OBSERVABLE_RUN_ID")
        or (
            datetime.now(timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid4().hex[:8]
        )
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(args.output_dir.iterdir()):
        raise RuntimeError(
            "Output directory is not empty: "
            f"{args.output_dir}"
        )

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(
            os.environ["NEO4J_USER"],
            os.environ["NEO4J_PASSWORD"],
        ),
    )

    lifecycle_store = None
    retriever = None

    try:
        driver.verify_connectivity()

        bootstrap = Neo4jSchemaBootstrap(
            driver=driver,
            database=database,
        )

        bootstrap.apply(
            wait_seconds=60
        )

        lifecycle_store = (
            Neo4jLifecycleStore(
                driver=driver,
                database=database,
            )
        )

        service = LifecycleService(
            lifecycle_store
        )

        retriever = Neo4jMemoryStore(
            driver=driver,
            database=database,
            config=config,
        )

        results: list[
            dict[str, Any]
        ] = []

        print(
            "========================================"
        )
        print(
            "PHASE II-B5 C3 OBSERVABLE RETRIEVAL"
        )
        print(
            "========================================"
        )
        print("Run ID:", run_id)
        print("Case count:", len(cases))
        print(
            "Observable stage:",
            "neo4j_semantic_retrieval_top20_role_conditioned",
        )
        print(
            "Database:",
            database or "<default>",
        )

        for index, case in enumerate(
            cases,
            start=1,
        ):
            row = run_case(
                case=case,
                run_id=run_id,
                service=service,
                lifecycle_store=(
                    lifecycle_store
                ),
                retriever=retriever,
                driver=driver,
                database=database,
            )

            results.append(row)

            checkpoint(
                output_dir=args.output_dir,
                rows=results,
                total_cases=len(cases),
            )

            print(
                f"[{index}/{len(cases)}] "
                f"{row['case_id']} "
                f"{row['pattern']} "
                f"current={len(row['current_search_texts'])} "
                f"historical={len(row['historical_search_texts'])}"
            )

        summary = {
            "experiment":
                "c3_lifecycle_observable_retrieval",

            "run_id": run_id,

            "case_count":
                len(results),

            "observable_stage":
                "neo4j_semantic_retrieval_top20_role_conditioned",

            "memory_formation":
                "C3 LifecycleService",

            "persistence":
                "Neo4jLifecycleStore",

            "retrieval":
                "direct Neo4jMemoryStore semantic retrieval",

            "generation":
                "MockBackbone; generation output "
                "is not evaluated",

            "final_memory_scope":
                "semantic lifecycle records",

            "pipeline_checks": {
                "all_cases_completed": (
                    len(results)
                    == len(cases)
                ),

                "all_current_searches_nonempty":
                    all(
                        bool(
                            row[
                                "current_search_texts"
                            ]
                        )
                        for row in results
                    ),

                "all_historical_searches_nonempty":
                    all(
                        bool(
                            row[
                                "historical_search_texts"
                            ]
                        )
                        for row in results
                    ),

                "operation_sequences_validated":
                    True,
            },
        }

        (
            args.output_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        config_snapshot = {
            "cases": str(
                args.cases.resolve()
            ),
            "config": str(
                args.config.resolve()
            ),
            "surface_variant":
                args.surface_variant,
            "state_key":
                args.state_key,
            "pattern":
                args.pattern,
            "limit":
                args.limit,
            "run_id":
                run_id,
            "database":
                database,
            "observable_stage":
                "neo4j_semantic_retrieval_top20_role_conditioned",
        }

        (
            args.output_dir
            / "config.json"
        ).write_text(
            json.dumps(
                config_snapshot,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        if not all(
            summary[
                "pipeline_checks"
            ].values()
        ):
            raise RuntimeError(
                "Observable retrieval pipeline "
                "checks failed."
            )

        print()
        print(
            "C3 OBSERVABLE RETRIEVAL: PASSED"
        )

    finally:
        if retriever is not None:
            try:
                retriever.close()
            except Exception:
                pass

        driver.close()


if __name__ == "__main__":
    main()
