from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def get_field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def load_memories(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
        return payload["memories"]
    raise TypeError("Unsupported memories JSON structure.")


def build_source_map(memories: list[dict[str, Any]], split_ids: Any) -> dict[str, list[str]]:
    source_map: dict[str, list[str]] = {}
    for record in memories:
        memory_id = str(get_field(record, "memory_id", "")).strip()
        memory_type = str(get_field(record, "memory_type", "")).strip()
        source_ids = split_ids(get_field(record, "source_ids", []))
        if not memory_id:
            continue
        source_map[memory_id] = (
            unique(source_ids)
            if memory_type == "semantic" and source_ids
            else [memory_id]
        )
    return source_map


def project_ids(memory_ids: list[str], source_map: dict[str, list[str]]) -> list[str]:
    projected: list[str] = []
    for memory_id in memory_ids:
        projected.extend(source_map.get(memory_id, [memory_id]))
    return unique(projected)


def source_equivalent_rescore(
    *,
    runner: Any,
    questions: list[dict[str, str]],
    raw_rows: list[dict[str, Any]],
    memories_path: Path,
    output_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    source_map = build_source_map(
        load_memories(memories_path),
        runner.split_ids,
    )
    questions_by_id = {
        str(row["question_id"]): row
        for row in questions
    }
    raw_by_id = {
        str(row["question_id"]): row
        for row in raw_rows
    }

    if set(raw_by_id) != set(questions_by_id):
        raise AssertionError("Raw retrieval IDs do not align with questions.")

    output_rows: list[dict[str, Any]] = []

    for question in questions:
        question_id = str(question["question_id"])
        raw = raw_by_id[question_id]

        gold_memory_ids = runner.split_ids(
            question.get("supporting_memory_ids", "")
        )
        retrieved_memory_ids = runner.split_ids(
            raw.get("retrieved_memory_ids", [])
        )

        gold_source_ids = project_ids(gold_memory_ids, source_map)
        retrieved_source_ids = project_ids(retrieved_memory_ids, source_map)

        gold_set = set(gold_source_ids)
        retrieved_set = set(retrieved_source_ids)
        overlap = sorted(gold_set & retrieved_set)

        precision = (
            len(overlap) / len(retrieved_set)
            if retrieved_set
            else 0.0
        )
        recall = (
            len(overlap) / len(gold_set)
            if gold_set
            else 0.0
        )

        output_rows.append(
            {
                **raw,
                "gold_memory_ids": gold_memory_ids,
                "gold_source_ids": gold_source_ids,
                "retrieved_source_ids": retrieved_source_ids,
                "projected_source_ids": retrieved_source_ids,
                "overlap_source_ids": overlap,
                "precision": precision,
                "recall": recall,
                "f1": harmonic_f1(precision, recall),
                "hit_at_k": bool(overlap),
                "retrieved_memory_count": len(retrieved_memory_ids),
                "projected_source_count": len(retrieved_source_ids),
                "scoring_mode": "source_equivalent_v02",
            }
        )

    source_dir = output_dir / "source_equiv_v02"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        source_dir
        / f"retrieval_top{top_k}_source_equiv.jsonl"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "top_k": top_k,
        "questions": len(output_rows),
        "scoring_mode": "source_equivalent_v02",
        "mean_precision": sum(row["precision"] for row in output_rows)
        / len(output_rows),
        "mean_recall": sum(row["recall"] for row in output_rows)
        / len(output_rows),
        "mean_f1": sum(row["f1"] for row in output_rows)
        / len(output_rows),
        "hit_at_k": sum(int(row["hit_at_k"]) for row in output_rows)
        / len(output_rows),
        "mean_retrieved_memory_count": sum(
            row["retrieved_memory_count"] for row in output_rows
        )
        / len(output_rows),
        "mean_projected_source_count": sum(
            row["projected_source_count"] for row in output_rows
        )
        / len(output_rows),
        "empty_retrievals": sum(
            int(row["retrieved_memory_count"] == 0)
            for row in output_rows
        ),
        "output_path": str(output_path),
    }

    (
        source_dir / f"summary_top{top_k}_source_equiv.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query existing provenance-preserving Mem0 LoCoMo collections "
            "on an arbitrary split and produce source-equivalent scores."
        )
    )
    parser.add_argument("--runner-module", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-questions", type=int, default=0)
    parser.add_argument(
        "--dataset-role",
        default="post-development held-out-from-pilot",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    runner = load_module(
        args.runner_module,
        "mem0_pilot_runner_module",
    )
    questions = read_csv(args.questions)

    if not questions:
        raise ValueError("Question file is empty.")

    question_ids = [str(row["question_id"]) for row in questions]

    if len(set(question_ids)) != len(question_ids):
        raise AssertionError("Question IDs are not unique.")

    if (
        args.expected_questions > 0
        and len(questions) != args.expected_questions
    ):
        raise AssertionError(
            f"Expected {args.expected_questions} questions, "
            f"found {len(questions)}."
        )

    required_fields = {
        "question_id",
        "user_id",
        "question_type",
        "question",
        "supporting_memory_ids",
    }
    missing_fields = required_fields - set(questions[0])

    if missing_fields:
        raise AssertionError(
            f"Question CSV is missing fields: {sorted(missing_fields)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"retrieval_top{args.top_k}.jsonl"

    existing_rows = read_jsonl(raw_path)
    existing_by_id = {
        str(row["question_id"]): row
        for row in existing_rows
    }

    unexpected = set(existing_by_id) - set(question_ids)
    if unexpected:
        raise AssertionError(
            f"Existing output has unexpected IDs: {sorted(unexpected)[:10]}"
        )

    questions_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for question in questions:
        questions_by_user[str(question["user_id"])].append(question)

    completed = set(existing_by_id)
    total = len(questions)

    print(
        f"Resume state: {len(completed)}/{total} questions complete.",
        flush=True,
    )

    with raw_path.open("a", encoding="utf-8") as handle:
        processed_now = 0

        for user_id in sorted(questions_by_user):
            store = runner.create_memory(user_id)
            try:
                for question in questions_by_user[user_id]:
                    question_id = str(question["question_id"])
                    if question_id in completed:
                        continue

                    started = time.perf_counter()
                    response = store.search(
                        question["question"],
                        top_k=args.top_k,
                        filters={"user_id": user_id},
                        threshold=0.0,
                        rerank=False,
                    )
                    latency = time.perf_counter() - started
                    results = runner.unwrap_results(response)

                    retrieved_memory_ids: list[str] = []
                    projected_source_ids: list[str] = []

                    for result in results:
                        metadata = runner.get_metadata(result)
                        conversation_id = str(
                            metadata.get("conversation_id", "")
                        ).strip()

                        if conversation_id and conversation_id != user_id:
                            raise AssertionError(
                                "Cross-conversation leakage: "
                                f"expected {user_id}, received "
                                f"{conversation_id}."
                            )

                        memory_id = runner.source_memory_id(result)
                        if memory_id:
                            retrieved_memory_ids.append(memory_id)

                        projected_source_ids.extend(
                            runner.project_result_to_sources(result)
                        )

                    retrieved_memory_ids = unique(retrieved_memory_ids)
                    projected_source_ids = unique(projected_source_ids)
                    gold_ids = runner.split_ids(
                        question.get("supporting_memory_ids", "")
                    )
                    overlap = sorted(
                        set(gold_ids) & set(projected_source_ids)
                    )

                    precision = (
                        len(overlap) / len(projected_source_ids)
                        if projected_source_ids
                        else 0.0
                    )
                    recall = (
                        len(overlap) / len(gold_ids)
                        if gold_ids
                        else 0.0
                    )

                    row = {
                        "question_id": question_id,
                        "user_id": user_id,
                        "question_type": question["question_type"],
                        "question": question["question"],
                        "gold_ids": gold_ids,
                        "retrieved_memory_ids": retrieved_memory_ids,
                        "projected_source_ids": projected_source_ids,
                        "overlap": overlap,
                        "precision": precision,
                        "recall": recall,
                        "f1": harmonic_f1(precision, recall),
                        "hit_at_k": bool(overlap),
                        "retrieved_memory_count": len(retrieved_memory_ids),
                        "projected_source_count": len(projected_source_ids),
                        "latency_seconds": latency,
                    }

                    append_jsonl(handle, row)
                    existing_by_id[question_id] = row
                    completed.add(question_id)
                    processed_now += 1

                    if (
                        len(completed) % 25 == 0
                        or len(completed) == total
                    ):
                        print(
                            f"top-{args.top_k}: "
                            f"{len(completed)}/{total}",
                            flush=True,
                        )
            finally:
                del store

    if len(completed) != total:
        raise AssertionError(
            f"Incomplete retrieval: {len(completed)}/{total}"
        )

    ordered_rows = [
        existing_by_id[question_id]
        for question_id in question_ids
    ]

    with raw_path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    raw_summary = {
        "top_k": args.top_k,
        "questions": len(ordered_rows),
        "dataset_role": args.dataset_role,
        "reused_collection_prefix": getattr(
            runner,
            "PILOT_NAME",
            "unknown",
        ),
        "mean_precision": sum(row["precision"] for row in ordered_rows)
        / len(ordered_rows),
        "mean_recall": sum(row["recall"] for row in ordered_rows)
        / len(ordered_rows),
        "mean_f1": sum(row["f1"] for row in ordered_rows)
        / len(ordered_rows),
        "hit_at_k": sum(int(row["hit_at_k"]) for row in ordered_rows)
        / len(ordered_rows),
        "mean_retrieved_memory_count": sum(
            row["retrieved_memory_count"] for row in ordered_rows
        )
        / len(ordered_rows),
        "mean_projected_source_count": sum(
            row["projected_source_count"] for row in ordered_rows
        )
        / len(ordered_rows),
        "mean_latency_seconds": sum(
            row["latency_seconds"] for row in ordered_rows
        )
        / len(ordered_rows),
        "empty_retrievals": sum(
            int(row["retrieved_memory_count"] == 0)
            for row in ordered_rows
        ),
        "processed_in_this_run": processed_now,
    }

    (
        args.output_dir / f"summary_top{args.top_k}.json"
    ).write_text(
        json.dumps(raw_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_summary = source_equivalent_rescore(
        runner=runner,
        questions=questions,
        raw_rows=ordered_rows,
        memories_path=args.memories,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    manifest = {
        "experiment": "mem0_locomo_heldout_query",
        "dataset_role": args.dataset_role,
        "questions": len(questions),
        "top_k": args.top_k,
        "collection_prefix": getattr(
            runner,
            "PILOT_NAME",
            "unknown",
        ),
        "ingestion_reused": True,
        "raw_summary": raw_summary,
        "source_equivalent_summary": source_summary,
    }

    (
        args.output_dir / "retrieval_manifest.json"
    ).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if raw_summary["empty_retrievals"] != 0:
        raise AssertionError("Raw retrieval contains empty results.")

    if source_summary["empty_retrievals"] != 0:
        raise AssertionError(
            "Source-equivalent retrieval contains empty results."
        )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print()
    print("MEM0 HELD-OUT RETRIEVAL: PASSED")


if __name__ == "__main__":
    main()
