#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mem0 import Memory
from openai import OpenAI


USER_ID = "phase2a_residence_smoke_v01"
MODEL_ID = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
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


def result_items(value: Any) -> list[dict[str, Any]]:
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
    texts: list[str] = []

    for item in result_items(value):
        text = (
            item.get("memory")
            or item.get("data")
            or item.get("text")
        )

        if isinstance(text, str):
            texts.append(text)

    return texts


def memory_ids(value: Any) -> list[str]:
    ids: list[str] = []

    for item in result_items(value):
        memory_id = item.get("id")

        if memory_id is not None:
            ids.append(str(memory_id))

    return ids


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

    print(
        "========================================"
    )
    print(
        "MEM0 INFER=TRUE LIFECYCLE SMOKE"
    )
    print(
        "========================================"
    )
    print("User ID:", USER_ID)
    print("Base URL:", args.base_url)

    ###########################################################################
    # Direct OpenAI-compatible JSON protocol probe
    ###########################################################################

    protocol_client = OpenAI(
        api_key="local-placeholder",
        base_url=args.base_url,
    )

    protocol_started = time.perf_counter()

    protocol_response = (
        protocol_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return a JSON object with "
                        'exactly {"status":"READY"}.'
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

    protocol_seconds = (
        time.perf_counter()
        - protocol_started
    )

    protocol_content = (
        protocol_response
        .choices[0]
        .message
        .content
    )

    protocol_value = json.loads(
        protocol_content
    )

    if protocol_value.get("status") != "READY":
        raise RuntimeError(
            "Unexpected protocol probe response: "
            f"{protocol_value!r}"
        )

    print()
    print(
        "OpenAI JSON protocol probe: PASSED"
    )
    print(
        "Protocol seconds:",
        round(protocol_seconds, 3),
    )
    print(
        "Protocol content:",
        protocol_content,
    )

    ###########################################################################
    # Mem0 configuration
    ###########################################################################

    config = {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": MODEL_ID,
                "api_key": (
                    "local-placeholder"
                ),
                "vllm_base_url": (
                    args.base_url
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
                    "phase2a_mem0_"
                    "lifecycle_smoke_v01"
                ),
                "embedding_model_dims": 384,
                "path": str(vector_path),
                "on_disk": True,
            },
        },
        "history_db_path": (
            str(history_path)
        ),
    }

    write_json(
        args.output_dir / "config.json",
        config,
    )

    memory = Memory.from_config(config)

    print()
    print(
        "Memory class:",
        type(memory).__name__,
    )
    print(
        "LLM class:",
        type(memory.llm).__name__,
    )
    print(
        "Vector store:",
        type(memory.vector_store).__name__,
    )

    turns = [
        {
            "turn_id": "turn_01_initial",
            "timestamp": (
                "2026-06-01T10:00:00+01:00"
            ),
            "content": (
                "I live in Nottingham."
            ),
            "expected_role": "initial_add",
        },
        {
            "turn_id": "turn_02_duplicate",
            "timestamp": (
                "2026-06-02T10:00:00+01:00"
            ),
            "content": (
                "I live in Nottingham."
            ),
            "expected_role": (
                "duplicate_or_noop"
            ),
        },
        {
            "turn_id": "turn_03_move",
            "timestamp": (
                "2026-07-15T10:00:00+01:00"
            ),
            "content": (
                "In July 2026, I moved "
                "from Nottingham to London."
            ),
            "expected_role": (
                "state_transition"
            ),
        },
        {
            "turn_id": "turn_04_current",
            "timestamp": (
                "2026-07-16T10:00:00+01:00"
            ),
            "content": (
                "I currently live in London."
            ),
            "expected_role": (
                "current_state_confirmation"
            ),
        },
    ]

    events: list[dict[str, Any]] = []

    for index, turn in enumerate(
        turns,
        start=1,
    ):
        print()
        print(
            f"===== TURN {index} ====="
        )
        print("Input:", turn["content"])

        started = time.perf_counter()

        add_result = memory.add(
            [
                {
                    "role": "user",
                    "content": (
                        turn["content"]
                    ),
                }
            ],
            user_id=USER_ID,
            metadata={
                "experiment": (
                    "phase2a_mem0_"
                    "lifecycle_smoke_v01"
                ),
                "turn_id": (
                    turn["turn_id"]
                ),
                "expected_role": (
                    turn["expected_role"]
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
                "user_id": USER_ID
            },
            top_k=100,
        )

        current_ids = memory_ids(snapshot)

        histories: dict[str, Any] = {}

        for memory_id in current_ids:
            histories[memory_id] = (
                memory.history(memory_id)
            )

        event = {
            "turn": turn,
            "add_result": add_result,
            "elapsed_seconds": elapsed,
            "snapshot": snapshot,
            "histories": histories,
            "memory_count": len(
                current_ids
            ),
            "memory_texts": (
                memory_texts(snapshot)
            ),
        }

        events.append(event)

        print(
            "Elapsed seconds:",
            round(elapsed, 3),
        )
        print(
            "Add result:",
            json.dumps(
                jsonable(add_result),
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(
            "Memory count:",
            len(current_ids),
        )
        print(
            "Memory texts:",
            memory_texts(snapshot),
        )

        write_json(
            args.output_dir
            / f"turn_{index:02d}.json",
            event,
        )

    ###########################################################################
    # Final retrieval checks
    ###########################################################################

    final_snapshot = memory.get_all(
        filters={"user_id": USER_ID},
        top_k=100,
    )

    current_search = memory.search(
        "Where does the user currently live?",
        filters={"user_id": USER_ID},
        top_k=10,
        threshold=0.0,
    )

    historical_search = memory.search(
        (
            "Where did the user live "
            "before moving to London?"
        ),
        filters={"user_id": USER_ID},
        top_k=10,
        threshold=0.0,
    )

    final_ids = memory_ids(
        final_snapshot
    )

    final_histories = {
        memory_id: memory.history(
            memory_id
        )
        for memory_id in final_ids
    }

    all_final_text = " ".join(
        memory_texts(final_snapshot)
    ).lower()

    current_text = " ".join(
        memory_texts(current_search)
    ).lower()

    historical_text = " ".join(
        memory_texts(historical_search)
    ).lower()

    summary = {
        "experiment": (
            "phase2a_mem0_"
            "lifecycle_smoke_v01"
        ),
        "user_id": USER_ID,
        "turn_count": len(turns),
        "protocol_probe_passed": True,
        "final_memory_count": len(
            final_ids
        ),
        "final_memory_ids": final_ids,
        "final_memory_texts": (
            memory_texts(final_snapshot)
        ),
        "current_search": current_search,
        "historical_search": (
            historical_search
        ),
        "final_histories": (
            final_histories
        ),
        "behavioural_indicators": {
            "final_snapshot_mentions_london": (
                "london" in all_final_text
            ),
            "final_snapshot_mentions_nottingham": (
                "nottingham"
                in all_final_text
            ),
            "current_search_mentions_london": (
                "london" in current_text
            ),
            "historical_search_mentions_nottingham": (
                "nottingham"
                in historical_text
            ),
        },
        "pipeline_checks": {
            "all_turns_completed": (
                len(events) == 4
            ),
            "final_memory_nonempty": (
                len(final_ids) > 0
            ),
            "current_search_nonempty": (
                len(
                    result_items(
                        current_search
                    )
                )
                > 0
            ),
            "historical_search_nonempty": (
                len(
                    result_items(
                        historical_search
                    )
                )
                > 0
            ),
        },
    }

    write_json(
        args.output_dir / "events.json",
        events,
    )

    write_json(
        args.output_dir / "summary.json",
        summary,
    )

    print()
    print(
        "===== FINAL SUMMARY ====="
    )
    print(
        json.dumps(
            jsonable(summary),
            ensure_ascii=False,
            indent=2,
        )
    )

    strict_checks = (
        summary["pipeline_checks"]
    )

    if not all(
        strict_checks.values()
    ):
        raise RuntimeError(
            "One or more pipeline checks failed: "
            f"{strict_checks}"
        )

    print()
    print(
        "MEM0 INFER=TRUE PIPELINE SMOKE: PASSED"
    )
    print(
        "Lifecycle behaviour still requires "
        "manual and benchmark-level evaluation."
    )


if __name__ == "__main__":
    main()
