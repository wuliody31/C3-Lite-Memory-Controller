from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from mem0 import Memory

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from mem0_raw_backend import VECTOR_ROOT, build_raw_backend_config  # noqa: E402

PILOT_NAME = "rc88_locomo_pilot200_mem0_v01"
ALLOWED_MEMORY_TYPES = {"episodic", "semantic"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_memories(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
        return payload["memories"]
    raise TypeError("Unsupported memories JSON structure.")


def get_field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(dict.fromkeys(str(x).strip() for x in value if str(x).strip()))
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return split_ids(decoded)
        except json.JSONDecodeError:
            pass
    return list(dict.fromkeys(x.strip() for x in text.split(";") if x.strip()))


def database_name(user_id: str) -> str:
    return f"{PILOT_NAME}_{user_id}"


def collection_name(user_id: str) -> str:
    return f"{PILOT_NAME}_{user_id}"


def create_memory(user_id: str) -> Memory:
    return Memory.from_config(
        build_raw_backend_config(
            collection_name=collection_name(user_id),
            database_name=database_name(user_id),
        )
    )


def unwrap_results(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict) and isinstance(response.get("results", []), list):
        return response.get("results", [])
    raise TypeError(f"Unexpected Mem0 response type: {type(response)!r}")


def get_metadata(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("metadata")
    return value if isinstance(value, dict) else {}


def source_memory_id(result: dict[str, Any]) -> str:
    metadata = get_metadata(result)
    return str(metadata.get("source_memory_id", metadata.get("source_id", ""))).strip()


def project_result_to_sources(result: dict[str, Any]) -> list[str]:
    metadata = get_metadata(result)
    memory_id = source_memory_id(result)
    memory_type = str(metadata.get("memory_type", "")).strip()
    source_ids = split_ids(metadata.get("source_ids"))
    if memory_type == "semantic" and source_ids:
        return source_ids
    if memory_id:
        return [memory_id]
    return source_ids


def harmonic_f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def prepare_metadata(record: dict[str, Any], chronological_index: int) -> dict[str, Any]:
    memory_id = str(get_field(record, "memory_id", "")).strip()
    return {
        "source_id": memory_id,
        "source_memory_id": memory_id,
        "source_ids": split_ids(get_field(record, "source_ids", [])),
        "memory_type": str(get_field(record, "memory_type", "")).strip(),
        "source_timestamp": str(get_field(record, "timestamp", "")).strip(),
        "conversation_id": str(get_field(record, "user_id", "")).strip(),
        "chronological_index": chronological_index,
        "backend_mode": "raw_infer_false",
    }


def ingest(*, memories_path: Path, questions_path: Path, output_dir: Path) -> None:
    memories = load_memories(memories_path)
    questions = read_csv(questions_path)
    pilot_users = sorted({row["user_id"] for row in questions})
    memories_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in memories:
        user_id = str(get_field(record, "user_id", "")).strip()
        memory_type = str(get_field(record, "memory_type", "")).strip()
        if user_id in pilot_users and memory_type in ALLOWED_MEMORY_TYPES:
            memories_by_user[user_id].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_conversation = []
    total_started = time.perf_counter()

    for user_index, user_id in enumerate(pilot_users, start=1):
        records = memories_by_user[user_id]
        if not records:
            raise ValueError(f"No memories found for {user_id}.")

        records.sort(
            key=lambda record: (
                str(get_field(record, "timestamp", "")),
                str(get_field(record, "memory_type", "")),
                str(get_field(record, "memory_id", "")),
            )
        )

        db_path = VECTOR_ROOT / database_name(user_id)
        if db_path.exists():
            shutil.rmtree(db_path)

        store = create_memory(user_id)
        started = time.perf_counter()
        seen: set[str] = set()
        episodic_count = 0
        semantic_count = 0

        for index, record in enumerate(records, start=1):
            memory_id = str(get_field(record, "memory_id", "")).strip()
            text = str(get_field(record, "text", "")).strip()
            memory_type = str(get_field(record, "memory_type", "")).strip()

            if not memory_id:
                raise ValueError(f"Missing memory_id for {user_id}, row {index}.")
            if not text:
                raise ValueError(f"Missing memory text for {memory_id}.")
            if memory_id in seen:
                raise AssertionError(f"Duplicate memory ID: {memory_id}")
            seen.add(memory_id)

            store.add(
                text,
                user_id=user_id,
                metadata=prepare_metadata(record, index),
                infer=False,
            )

            episodic_count += int(memory_type == "episodic")
            semantic_count += int(memory_type == "semantic")

            if index % 100 == 0 or index == len(records):
                print(f"[{user_index}/{len(pilot_users)}] {user_id}: {index}/{len(records)}")

        row = {
            "user_id": user_id,
            "database_name": database_name(user_id),
            "collection_name": collection_name(user_id),
            "total_memories": len(records),
            "episodic_memories": episodic_count,
            "semantic_memories": semantic_count,
            "unique_memory_ids": len(seen),
            "ingestion_seconds": time.perf_counter() - started,
        }
        per_conversation.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del store

    summary = {
        "pilot_name": PILOT_NAME,
        "pilot_questions": len(questions),
        "conversations": len(pilot_users),
        "total_memories": sum(x["total_memories"] for x in per_conversation),
        "total_episodic_memories": sum(x["episodic_memories"] for x in per_conversation),
        "total_semantic_memories": sum(x["semantic_memories"] for x in per_conversation),
        "total_ingestion_seconds": time.perf_counter() - total_started,
        "per_conversation": per_conversation,
    }

    (output_dir / "ingestion_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("MEM0 PILOT200 INGESTION: PASSED")


def evaluate_at_k(
    *,
    questions: list[dict[str, str]],
    output_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    questions_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for question in questions:
        questions_by_user[question["user_id"]].append(question)

    result_by_id: dict[str, dict[str, Any]] = {}
    processed = 0

    for user_id in sorted(questions_by_user):
        store = create_memory(user_id)

        for question in questions_by_user[user_id]:
            started = time.perf_counter()
            response = store.search(
                question["question"],
                top_k=top_k,
                filters={"user_id": user_id},
                threshold=0.0,
                rerank=False,
            )
            latency = time.perf_counter() - started
            results = unwrap_results(response)

            retrieved_memory_ids: list[str] = []
            projected_source_ids: list[str] = []

            for result in results:
                metadata = get_metadata(result)
                conversation_id = str(metadata.get("conversation_id", "")).strip()
                if conversation_id and conversation_id != user_id:
                    raise AssertionError(
                        f"Cross-conversation leakage: expected {user_id}, received {conversation_id}."
                    )

                memory_id = source_memory_id(result)
                if memory_id:
                    retrieved_memory_ids.append(memory_id)
                projected_source_ids.extend(project_result_to_sources(result))

            retrieved_memory_ids = list(dict.fromkeys(retrieved_memory_ids))
            projected_source_ids = list(dict.fromkeys(projected_source_ids))
            gold_ids = split_ids(question.get("supporting_memory_ids", ""))
            gold_set = set(gold_ids)
            projected_set = set(projected_source_ids)
            overlap = gold_set & projected_set

            precision = len(overlap) / len(projected_set) if projected_set else 0.0
            recall = len(overlap) / len(gold_set) if gold_set else 0.0
            f1 = harmonic_f1(precision, recall)

            result_by_id[question["question_id"]] = {
                "question_id": question["question_id"],
                "user_id": user_id,
                "question_type": question["question_type"],
                "question": question["question"],
                "gold_ids": gold_ids,
                "retrieved_memory_ids": retrieved_memory_ids,
                "projected_source_ids": projected_source_ids,
                "overlap": sorted(overlap),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit_at_k": bool(overlap),
                "retrieved_memory_count": len(retrieved_memory_ids),
                "projected_source_count": len(projected_source_ids),
                "latency_seconds": latency,
            }

            processed += 1
            if processed % 25 == 0 or processed == len(questions):
                print(f"top-{top_k}: {processed}/{len(questions)}")

        del store

    rows = [result_by_id[q["question_id"]] for q in questions]
    output_path = output_dir / f"retrieval_top{top_k}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "top_k": top_k,
        "questions": len(rows),
        "mean_precision": sum(x["precision"] for x in rows) / len(rows),
        "mean_recall": sum(x["recall"] for x in rows) / len(rows),
        "mean_f1": sum(x["f1"] for x in rows) / len(rows),
        "hit_at_k": sum(int(x["hit_at_k"]) for x in rows) / len(rows),
        "mean_retrieved_memory_count": sum(x["retrieved_memory_count"] for x in rows) / len(rows),
        "mean_projected_source_count": sum(x["projected_source_count"] for x in rows) / len(rows),
        "mean_latency_seconds": sum(x["latency_seconds"] for x in rows) / len(rows),
        "empty_retrievals": sum(int(x["retrieved_memory_count"] == 0) for x in rows),
    }

    (output_dir / f"summary_top{top_k}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def query(*, questions_path: Path, output_dir: Path) -> None:
    questions = read_csv(questions_path)
    if len(questions) != 200:
        raise ValueError(f"Pilot file must contain 200 questions; found {len(questions)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        f"top{k}": evaluate_at_k(
            questions=questions,
            output_dir=output_dir,
            top_k=k,
        )
        for k in (5, 20)
    }

    combined = {
        "pilot_name": PILOT_NAME,
        "pilot_questions": len(questions),
        "configurations": summaries,
    }
    (output_dir / "retrieval_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for name, summary in summaries.items():
        if summary["questions"] != 200:
            raise AssertionError(f"{name} question count is not 200.")
        if summary["empty_retrievals"] != 0:
            raise AssertionError(f"{name} contains empty retrievals.")

    print()
    print("MEM0 PILOT200 RETRIEVAL: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run provenance-preserving Mem0 retrieval on the fixed LoCoMo Pilot-200 set."
    )
    parser.add_argument("--phase", choices=("ingest", "query"), required=True)
    parser.add_argument("--memories", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.phase == "ingest":
        if args.memories is None:
            parser.error("--memories is required for ingest.")
        ingest(
            memories_path=args.memories,
            questions_path=args.questions,
            output_dir=args.output_dir,
        )
        return

    query(
        questions_path=args.questions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
