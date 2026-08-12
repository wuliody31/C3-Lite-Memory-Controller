from __future__ import annotations

from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import QueryState


ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> C3Pipeline:
    config = load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )

    stopwords = set(
        config["query_analysis"]["stopwords"]
    )

    memory_store = (
        InMemoryMemoryStore.from_json_file(
            str(
                ROOT
                / "examples"
                / "demo_memories.json"
            ),
            stopwords,
        )
    )

    procedure_store = (
        ProceduralJsonStore.from_file(
            str(
                ROOT
                / "examples"
                / "demo_procedures.json"
            ),
            stopwords,
        )
    )

    return C3Pipeline(
        config=config,
        memory_store=memory_store,
        procedure_store=procedure_store,
        backbone=MockBackbone(),
        prompt_template=(
            ROOT
            / "prompts"
            / "answer_prompt.txt"
        ),
    )


def print_candidate(
    memory_id: str,
    item: dict,
) -> None:
    print(
        memory_id,
        "| old_validity =",
        item["old_validity_score"],
        "| old_temporal =",
        item["old_temporal_task_score"],
        "| new_validity =",
        item[
            "new_query_relative_validity"
        ],
        "| role =",
        item["temporal_role"],
        "| compatible =",
        item["compatible"],
        "| status =",
        item["status"],
        "| action =",
        item["resolution_action"],
    )


def main() -> None:
    pipeline = build_pipeline()

    queries = [
        (
            "CURRENT",
            "What is my current MSc project scope?",
        ),
        (
            "HISTORICAL",
            "What was my previous MSc project scope?",
        ),
        (
            "TIMELINE",
            "How did my project scope change over time?",
        ),
    ]

    try:
        for label, query in queries:
            result = pipeline.answer(
                QueryState(
                    query=query,
                    user_id="user01",
                )
            )

            print()
            print("=" * 100)
            print("CASE:", label)
            print("QUERY:", query)
            print(
                "QUERY MODE:",
                result.query_mode.value,
            )
            print(
                "FINAL SELECTED:",
                result.selected_ids,
            )

            print(
                "\nTEMPORAL VALIDITY TRACE:"
            )

            trace = result.debug[
                "c3_v3_temporal_validity"
            ]

            for memory_id, item in trace.items():
                print_candidate(
                    memory_id,
                    item,
                )

            print(
                "\nFOCUS COMPARISON:"
            )

            for memory_id in [
                "s_user01_003",
                "s_user01_004",
            ]:
                if memory_id not in trace:
                    print(
                        memory_id,
                        "NOT IN RESOLVED CANDIDATES",
                    )
                    continue

                item = trace[memory_id]

                print(
                    memory_id,
                    "old=",
                    item[
                        "old_validity_score"
                    ],
                    "new=",
                    item[
                        "new_query_relative_validity"
                    ],
                    "role=",
                    item[
                        "temporal_role"
                    ],
                )

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()