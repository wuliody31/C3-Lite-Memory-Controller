#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mem0 import Memory
from openai import OpenAI


MODEL_ID = "Qwen/Qwen3-8B"

PATTERNS = (
    "replacement",
    "duplicate",
    "multi_step",
    "out_of_order",
    "reversion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        required=True,
    )

    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--backend-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            jsonable(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):
        return jsonable(
            value.model_dump()
        )

    if hasattr(value, "__dict__"):
        return jsonable(
            vars(value)
        )

    return str(value)


def result_items(
    value: Any,
) -> list[dict[str, Any]]:
    normalized = jsonable(value)

    if isinstance(normalized, list):
        return [
            item
            for item in normalized
            if isinstance(item, dict)
        ]

    if isinstance(normalized, dict):
        for key in (
            "results",
            "memories",
            "data",
        ):
            items = normalized.get(key)

            if isinstance(items, list):
                return [
                    item
                    for item in items
                    if isinstance(item, dict)
                ]

    return []


def memory_texts(value: Any) -> list[str]:
    output = []

    for item in result_items(value):
        text = (
            item.get("memory")
            or item.get("data")
            or item.get("text")
        )

        if isinstance(text, str):
            output.append(text)

    return output


def memory_ids(value: Any) -> list[str]:
    output = []

    for item in result_items(value):
        memory_id = item.get("id")

        if memory_id is not None:
            output.append(
                str(memory_id)
            )

    return output


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_text(
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    jsonable(row),
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_pilot_cases(
    path: Path,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    selected = [
        row
        for row in rows
        if row["surface_variant"] == 1
    ]

    if len(selected) != 25:
        raise RuntimeError(
            "Expected 25 variant-1 pilot cases, "
            f"got {len(selected)}."
        )

    pattern_counts = {
        pattern: sum(
            row["pattern"] == pattern
            for row in selected
        )
        for pattern in PATTERNS
    }

    if set(pattern_counts.values()) != {5}:
        raise RuntimeError(
            "Pilot is not balanced by pattern: "
            f"{pattern_counts}"
        )

    state_keys = sorted(
        {
            row["state_key"]
            for row in selected
        }
    )

    state_counts = {
        state_key: sum(
            row["state_key"] == state_key
            for row in selected
        )
        for state_key in state_keys
    }

    if (
        len(state_keys) != 5
        or set(state_counts.values()) != {5}
    ):
        raise RuntimeError(
            "Pilot is not balanced by state key: "
            f"{state_counts}"
        )

    return selected

def build_config(
    *,
    base_url: str,
    vector_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    return {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": MODEL_ID,
                "api_key": (
                    "local-placeholder"
                ),
                "vllm_base_url": (
                    base_url
                ),
                "temperature": 0.0,
                "max_tokens": 512,
                "top_p": 0.1,
                "top_k": 1,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": (
                    "sentence-transformers/"
                    "multi-qa-MiniLM-L6-cos-v1"
                ),
                "embedding_dims": 384,
                "model_kwargs": {
                    "device": "cpu",
                    "cache_folder": (
                        "/data/alyjw80/"
                        "c3_lite/cache/"
                        "huggingface/"
                        "sentence_transformers"
                    ),
                },
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": (
                    "phase2b_b5_mem0_"
                    "lifecycle_pilot25_v01"
                ),
                "embedding_model_dims": 384,
                "path": str(
                    vector_path
                ),
                "on_disk": True,
            },
        },
        "history_db_path": str(
            history_path
        ),
    }


def protocol_probe(
    base_url: str,
) -> dict[str, Any]:
    client = OpenAI(
        api_key="local-placeholder",
        base_url=base_url,
    )

    started = time.perf_counter()

    response = (
        client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object "
                        "with exactly "
                        '{"status":"READY"}.'
                    ),
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.0,
            max_tokens=64,
        )
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    parsed = json.loads(content)

    if parsed.get("status") != "READY":
        raise RuntimeError(
            "Unexpected protocol response: "
            f"{parsed!r}"
        )

    return {
        "passed": True,
        "elapsed_seconds": elapsed,
        "content": content,
    }


def safe_histories(
    memory: Memory,
    snapshot: Any,
) -> dict[str, Any]:
    output = {}

    for memory_id in memory_ids(
        snapshot
    ):
        try:
            output[memory_id] = (
                memory.history(
                    memory_id
                )
            )
        except Exception as exc:
            output[memory_id] = {
                "history_error": (
                    type(exc).__name__
                ),
                "message": str(exc),
            }

    return output


def run_case(
    *,
    memory: Memory,
    case: dict[str, Any],
) -> dict[str, Any]:
    user_id = (
        "phase2b_b5_"
        + case["case_id"]
    )

    print()
    print("=" * 72)
    print("CASE:", case["case_id"])
    print("PATTERN:", case["pattern"])
    print("USER:", user_id)
    print("=" * 72)

    turn_rows = []

    for event in case["events"]:
        print()
        print(
            "EVENT:",
            event["event_id"],
            event["observed_at"],
        )
        print(
            "INPUT:",
            event["text"],
        )

        started = (
            time.perf_counter()
        )

        add_result = memory.add(
            [
                {
                    "role": "user",
                    "content": (
                        event["text"]
                    ),
                }
            ],
            user_id=user_id,
            metadata={
                "experiment": (
                    "phase2b_b5_mem0_"
                    "lifecycle_pilot25_v01"
                ),
                "case_id": (
                    case["case_id"]
                ),
                "pattern": (
                    case["pattern"]
                ),
                "event_id": (
                    event["event_id"]
                ),
                "ingestion_index": (
                    event[
                        "ingestion_index"
                    ]
                ),
                # Provenance only.
                # Mem0 OSS timestamp API
                # is not used here.
                "observed_at": (
                    event["observed_at"]
                ),
            },
            infer=True,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        snapshot = memory.get_all(
            filters={
                "user_id": user_id
            },
            top_k=100,
        )

        turn_row = {
            "event": event,
            "add_result": add_result,
            "elapsed_seconds": elapsed,
            "snapshot": snapshot,
            "memory_count": len(
                memory_ids(
                    snapshot
                )
            ),
            "memory_texts": (
                memory_texts(
                    snapshot
                )
            ),
            "histories": (
                safe_histories(
                    memory,
                    snapshot,
                )
            ),
        }

        turn_rows.append(
            turn_row
        )

        print(
            "Elapsed:",
            round(elapsed, 3),
        )

        print(
            "Add events:",
            [
                item.get("event")
                for item
                in result_items(
                    add_result
                )
            ],
        )

        print(
            "Memory count:",
            turn_row[
                "memory_count"
            ],
        )

        print(
            "Memory texts:",
            json.dumps(
                turn_row[
                    "memory_texts"
                ],
                ensure_ascii=False,
            ),
        )

    final_snapshot = memory.get_all(
        filters={
            "user_id": user_id
        },
        top_k=100,
    )

    current_search = memory.search(
        case["current_query"],
        filters={
            "user_id": user_id
        },
        top_k=20,
        threshold=0.0,
    )

    historical_search = memory.search(
        case["historical_query"],
        filters={
            "user_id": user_id
        },
        top_k=20,
        threshold=0.0,
    )

    result = {
        "case_id": case["case_id"],
        "pattern": case["pattern"],
        "state_key": (
            case["state_key"]
        ),
        "surface_variant": (
            case["surface_variant"]
        ),
        "mem0_user_id": user_id,
        "temporal_protocol": {
            "structured_timestamp_api": (
                False
            ),
            "observed_at_passed_as": (
                "metadata_provenance_only"
            ),
            "event_text_contains_date": (
                False
            ),
            "note": (
                "The evaluated Mem0 OSS "
                "interface rejects the "
                "timestamp parameter. "
                "Out-of-order therefore "
                "cannot be treated as a "
                "pure temporal-policy "
                "comparison in this run."
            ),
        },
        "gold": {
            "current": (
                case["current_gold"]
            ),
            "previous": (
                case[
                    "historical_gold"
                ]
            ),
            "full_history": (
                case[
                    "history_values_expected"
                ]
            ),
            "stale_values": (
                case["stale_values"]
            ),
        },
        "current_query": (
            case["current_query"]
        ),
        "historical_query": (
            case[
                "historical_query"
            ]
        ),
        "turns": turn_rows,
        "final_snapshot": (
            final_snapshot
        ),
        "final_memory_ids": (
            memory_ids(
                final_snapshot
            )
        ),
        "final_memory_texts": (
            memory_texts(
                final_snapshot
            )
        ),
        "final_histories": (
            safe_histories(
                memory,
                final_snapshot,
            )
        ),
        "current_search": (
            current_search
        ),
        "current_search_texts": (
            memory_texts(
                current_search
            )
        ),
        "historical_search": (
            historical_search
        ),
        "historical_search_texts": (
            memory_texts(
                historical_search
            )
        ),
    }

    print()
    print(
        "FINAL MEMORIES:",
        json.dumps(
            result[
                "final_memory_texts"
            ],
            ensure_ascii=False,
        ),
    )

    print(
        "CURRENT SEARCH:",
        json.dumps(
            result[
                "current_search_texts"
            ],
            ensure_ascii=False,
        ),
    )

    print(
        "HISTORICAL SEARCH:",
        json.dumps(
            result[
                "historical_search_texts"
            ],
            ensure_ascii=False,
        ),
    )

    return result


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.backend_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_path = (
        args.backend_dir
        / "qdrant"
    )

    history_path = (
        args.backend_dir
        / "history.db"
    )

    vector_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_pilot_cases(
        args.cases
    )

    print(
        "========================================"
    )
    print(
        "PHASE II-B5 MEM0 25-CASE PILOT"
    )
    print(
        "========================================"
    )

    print(
        "Case count:",
        len(cases),
    )

    print(
        "Patterns:",
        [
            case["pattern"]
            for case in cases
        ],
    )

    probe = protocol_probe(
        args.base_url
    )

    print(
        "Protocol probe: PASSED"
    )

    config = build_config(
        base_url=args.base_url,
        vector_path=vector_path,
        history_path=history_path,
    )

    write_json(
        args.output_dir
        / "config.json",
        config,
    )

    memory = Memory.from_config(
        config
    )

    print(
        "Memory:",
        type(memory).__name__,
    )

    print(
        "LLM:",
        type(memory.llm).__name__,
    )

    print(
        "Vector store:",
        type(
            memory.vector_store
        ).__name__,
    )

    results = []

    results_path = (
        args.output_dir
        / "results.jsonl"
    )

    progress_path = (
        args.output_dir
        / "progress.json"
    )

    for case in cases:
        result = run_case(
            memory=memory,
            case=case,
        )

        results.append(result)

        # Persist after every completed case.
        # Infrastructure failures therefore
        # cannot erase already completed cases.
        write_jsonl(
            results_path,
            results,
        )

        write_json(
            progress_path,
            {
                "completed_cases": len(
                    results
                ),
                "total_cases": len(
                    cases
                ),
                "last_case_id": (
                    case["case_id"]
                ),
                "last_pattern": (
                    case["pattern"]
                ),
                "last_state_key": (
                    case["state_key"]
                ),
            },
        )

        print(
            "CHECKPOINT:",
            len(results),
            "/",
            len(cases),
            case["case_id"],
        )

    summary = {
        "experiment": (
            "phase2b_b5_mem0_"
            "lifecycle_pilot25_v01"
        ),
        "system": (
            "mem0_oss_infer_true"
        ),
        "model": MODEL_ID,
        "case_count": len(
            results
        ),
        "patterns": [
            row["pattern"]
            for row in results
        ],
        "protocol_probe": probe,
        "case_summaries": [
            {
                "case_id": (
                    row["case_id"]
                ),
                "pattern": (
                    row["pattern"]
                ),
                "final_memory_count": (
                    len(
                        row[
                            "final_memory_ids"
                        ]
                    )
                ),
                "final_memory_texts": (
                    row[
                        "final_memory_texts"
                    ]
                ),
                "current_search_texts": (
                    row[
                        "current_search_texts"
                    ]
                ),
                "historical_search_texts": (
                    row[
                        "historical_search_texts"
                    ]
                ),
            }
            for row in results
        ],
        "pipeline_checks": {
            "twenty_five_cases_completed": (
                len(results) == 25
            ),
            "all_final_snapshots_nonempty": (
                all(
                    len(
                        row[
                            "final_memory_ids"
                        ]
                    )
                    > 0
                    for row in results
                )
            ),
            "all_current_searches_nonempty": (
                all(
                    len(
                        result_items(
                            row[
                                "current_search"
                            ]
                        )
                    )
                    > 0
                    for row in results
                )
            ),
        },
        "interpretation_boundary": (
            "This is a balanced 25-case pilot "
            "and raw-behaviour capture, "
            "not a formal comparative "
            "evaluation. The out-of-order "
            "case is not temporally "
            "symmetric with C3 because "
            "the evaluated Mem0 OSS API "
            "does not accept structured "
            "observation timestamps."
        ),
    }

    write_json(
        args.output_dir
        / "summary.json",
        summary,
    )

    print()
    print(
        "===== SMOKE SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    if not all(
        summary[
            "pipeline_checks"
        ].values()
    ):
        raise RuntimeError(
            "One or more protocol "
            "checks failed."
        )

    print()
    print(
        "PHASE II-B5 MEM0 "
        "25-CASE PILOT: PASSED"
    )


if __name__ == "__main__":
    main()
