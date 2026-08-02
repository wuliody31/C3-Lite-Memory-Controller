from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from mem0 import Memory


SCRIPT_DIR = Path(
    __file__
).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

from mem0_raw_backend import (  # noqa: E402
    VECTOR_ROOT,
    build_raw_backend_config,
)


COLLECTION_NAME = (
    "rc88_locomo_conv30_raw_v01"
)

DATABASE_NAME = (
    "rc88_locomo_conv30_raw_v01"
)

CONVERSATION_ID = "locomo_conv_30"


def load_memories(
    path: Path,
) -> list[dict[str, Any]]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(payload, list):
        memories = payload
    elif (
        isinstance(payload, dict)
        and isinstance(
            payload.get("memories"),
            list,
        )
    ):
        memories = payload["memories"]
    else:
        raise TypeError(
            "Unsupported memories JSON "
            "structure."
        )

    return memories


def read_questions(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def create_memory() -> Memory:
    return Memory.from_config(
        build_raw_backend_config(
            collection_name=(
                COLLECTION_NAME
            ),
            database_name=(
                DATABASE_NAME
            ),
        )
    )


def unwrap_results(
    response: Any,
) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response

    if isinstance(response, dict):
        results = response.get(
            "results",
            [],
        )

        if isinstance(results, list):
            return results

    raise TypeError(
        "Unexpected Mem0 response: "
        f"{type(response)!r}"
    )


def extract_source_id(
    result: dict[str, Any],
) -> str | None:
    metadata = result.get(
        "metadata"
    )

    if isinstance(metadata, dict):
        value = metadata.get(
            "source_id"
        )

        if value:
            return str(value)

    value = result.get(
        "source_id"
    )

    return (
        str(value)
        if value
        else None
    )


def harmonic_f1(
    precision: float,
    recall: float,
) -> float:
    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def ingest(
    *,
    memories_path: Path,
    output_dir: Path,
) -> None:
    all_memories = load_memories(
        memories_path
    )

    episodic = [
        memory
        for memory in all_memories
        if memory.get("user_id")
        == CONVERSATION_ID
        and memory.get(
            "memory_type"
        )
        == "episodic"
    ]

    episodic.sort(
        key=lambda memory: (
            str(
                memory.get(
                    "timestamp",
                    "",
                )
            ),
            str(
                memory.get(
                    "memory_id",
                    "",
                )
            ),
        )
    )

    if not episodic:
        raise ValueError(
            "No episodic memories found "
            "for locomo_conv_30."
        )

    database_path = (
        VECTOR_ROOT
        / DATABASE_NAME
    )

    if database_path.exists():
        shutil.rmtree(
            database_path
        )

    memory_store = create_memory()

    started = time.perf_counter()

    added_ids: list[str] = []

    for index, record in enumerate(
        episodic,
        start=1,
    ):
        source_id = str(
            record["memory_id"]
        )

        text = str(
            record["text"]
        )

        source_timestamp = str(
            record.get(
                "timestamp",
                "",
            )
        )

        memory_store.add(
            text,
            user_id=CONVERSATION_ID,
            metadata={
                "source_id": source_id,
                "memory_type": (
                    "episodic"
                ),
                "source_timestamp": (
                    source_timestamp
                ),
                "conversation_id": (
                    CONVERSATION_ID
                ),
                "chronological_index": (
                    index
                ),
                "backend_mode": (
                    "raw_infer_false"
                ),
            },
            infer=False,
        )

        added_ids.append(
            source_id
        )

        if (
            index % 50 == 0
            or index == len(episodic)
        ):
            print(
                f"Ingested "
                f"{index}/{len(episodic)}"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "conversation_id": (
            CONVERSATION_ID
        ),
        "collection_name": (
            COLLECTION_NAME
        ),
        "database_name": (
            DATABASE_NAME
        ),
        "memory_mode": (
            "raw_infer_false"
        ),
        "memory_type": "episodic",
        "ingested_memories": len(
            episodic
        ),
        "unique_source_ids": len(
            set(added_ids)
        ),
        "ingestion_seconds": elapsed,
        "first_source_id": (
            added_ids[0]
        ),
        "last_source_id": (
            added_ids[-1]
        ),
    }

    if len(set(added_ids)) != len(
        added_ids
    ):
        raise AssertionError(
            "Duplicate episodic source IDs."
        )

    (
        output_dir
        / "ingestion_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print(
        "LOCOMO CONV30 INGESTION: "
        "PASSED"
    )


def query(
    *,
    questions_path: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    questions = read_questions(
        questions_path
    )

    if len(questions) != 10:
        raise ValueError(
            "Smoke set must contain "
            "exactly 10 questions."
        )

    memory_store = create_memory()

    per_question: list[
        dict[str, Any]
    ] = []

    prediction_path = (
        output_dir
        / "retrieval_results.jsonl"
    )

    with prediction_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for row in questions:
            started = (
                time.perf_counter()
            )

            response = memory_store.search(
                row["question"],
                top_k=top_k,
                filters={
                    "user_id": (
                        CONVERSATION_ID
                    ),
                },
                threshold=0.0,
                rerank=False,
            )

            latency = (
                time.perf_counter()
                - started
            )

            results = unwrap_results(
                response
            )

            retrieved_ids = [
                source_id
                for source_id in (
                    extract_source_id(
                        result
                    )
                    for result in results
                )
                if source_id
            ]

            retrieved_ids = list(
                dict.fromkeys(
                    retrieved_ids
                )
            )

            gold_ids = [
                value
                for value in row[
                    "supporting_memory_ids"
                ].split(";")
                if value
            ]

            gold_set = set(
                gold_ids
            )

            retrieved_set = set(
                retrieved_ids
            )

            overlap = (
                gold_set
                & retrieved_set
            )

            precision = (
                len(overlap)
                / len(retrieved_set)
                if retrieved_set
                else 0.0
            )

            recall = (
                len(overlap)
                / len(gold_set)
                if gold_set
                else 0.0
            )

            f1 = harmonic_f1(
                precision,
                recall,
            )

            cross_conversation = [
                source_id
                for source_id
                in retrieved_ids
                if not source_id.startswith(
                    "e_locomo_conv_30_"
                )
            ]

            if cross_conversation:
                raise AssertionError(
                    "Cross-conversation "
                    f"leakage: "
                    f"{cross_conversation}"
                )

            result_row = {
                "question_id": (
                    row["question_id"]
                ),
                "question_type": (
                    row["question_type"]
                ),
                "question": (
                    row["question"]
                ),
                "gold_ids": gold_ids,
                "retrieved_ids": (
                    retrieved_ids
                ),
                "overlap": sorted(
                    overlap
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit_at_k": bool(
                    overlap
                ),
                "retrieved_count": len(
                    retrieved_ids
                ),
                "latency_seconds": (
                    latency
                ),
            }

            per_question.append(
                result_row
            )

            output_handle.write(
                json.dumps(
                    result_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(
                row["question_id"],
                "| hit:",
                bool(overlap),
                "| P:",
                f"{precision:.3f}",
                "| R:",
                f"{recall:.3f}",
                "| F1:",
                f"{f1:.3f}",
            )

    summary = {
        "questions": len(
            per_question
        ),
        "top_k": top_k,
        "mean_precision": sum(
            row["precision"]
            for row in per_question
        )
        / len(per_question),
        "mean_recall": sum(
            row["recall"]
            for row in per_question
        )
        / len(per_question),
        "mean_f1": sum(
            row["f1"]
            for row in per_question
        )
        / len(per_question),
        "hit_at_k": sum(
            int(row["hit_at_k"])
            for row in per_question
        )
        / len(per_question),
        "mean_retrieved_count": sum(
            row["retrieved_count"]
            for row in per_question
        )
        / len(per_question),
        "mean_latency_seconds": sum(
            row["latency_seconds"]
            for row in per_question
        )
        / len(per_question),
        "empty_retrievals": sum(
            int(
                row[
                    "retrieved_count"
                ]
                == 0
            )
            for row in per_question
        ),
    }

    (
        output_dir
        / "retrieval_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    if summary[
        "empty_retrievals"
    ] != 0:
        raise AssertionError(
            "Smoke contains empty "
            "retrievals."
        )

    print(
        "LOCOMO CONV30 RETRIEVAL: "
        "PASSED"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        choices=(
            "ingest",
            "query",
        ),
        required=True,
    )
    parser.add_argument(
        "--memories",
        type=Path,
    )
    parser.add_argument(
        "--questions",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.phase == "ingest":
        if args.memories is None:
            parser.error(
                "--memories is required "
                "for ingest."
            )

        ingest(
            memories_path=(
                args.memories
            ),
            output_dir=(
                args.output_dir
            ),
        )
    else:
        if args.questions is None:
            parser.error(
                "--questions is required "
                "for query."
            )

        query(
            questions_path=(
                args.questions
            ),
            output_dir=(
                args.output_dir
            ),
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
