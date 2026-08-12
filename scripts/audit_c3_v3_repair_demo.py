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


def main() -> None:
    pipeline = build_pipeline()

    queries = [
        "What is my current MSc project scope?",
        "How did my project scope change over time?",
        "What was my previous MSc project scope?",
        "What procedure should I follow for writing the project report?",
        "Why did my MSc project scope change?",
    ]

    try:
        for query in queries:
            result = pipeline.answer(
                QueryState(
                    query=query,
                    user_id="user01",
                )
            )

            sufficiency = result.debug[
                "c3_v3_sufficiency"
            ]

            repair_plan = result.debug[
                "c3_v3_repair_plan"
            ]

            repair_execution = result.debug[
                "c3_v3_repair_execution"
            ]

            print("\n" + "=" * 80)
            print("QUERY:")
            print(query)

            print("\nFINAL SELECTED:")
            print(result.selected_ids)

            print("\nSUFFICIENCY:")
            print(
                "sufficient =",
                sufficiency["sufficient"],
            )
            print(
                "hard coverage =",
                sufficiency[
                    "hard_requirement_coverage"
                ],
            )
            print(
                "missing hard =",
                sufficiency[
                    "missing_hard_requirements"
                ],
            )

            print("\nREPAIR PLAN:")
            print(
                "needed =",
                repair_plan["needed"],
            )
            print(
                "missing roles =",
                repair_plan[
                    "missing_roles"
                ],
            )
            print(
                "targets =",
                repair_plan[
                    "target_memory_types"
                ],
            )

            print("\nREPAIR EXECUTION:")
            print(
                "attempted =",
                repair_execution[
                    "attempted"
                ],
            )
            print(
                "accepted =",
                repair_execution[
                    "accepted"
                ],
            )
            print(
                "retrieved =",
                repair_execution[
                    "repair_retrieved_ids"
                ],
            )
            print(
                "added =",
                repair_execution[
                    "added_candidate_ids"
                ],
            )
            print(
                "trace =",
                repair_execution[
                    "trace"
                ],
            )

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()