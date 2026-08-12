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
        config[
            "query_analysis"
        ][
            "stopwords"
        ]
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


def print_steps(
    steps: list[dict],
) -> None:
    if not steps:
        print("  <no arbitration steps>")
        return

    for step in steps:
        print(
            "  STEP",
            step.get("step"),
            "| phase=",
            step.get("phase"),
            "| memory=",
            step.get("memory_id"),
            "| hard_gain=",
            step.get("hard_gain"),
            "| soft_gain=",
            step.get("soft_gain"),
            "| requirement_gain=",
            step.get("requirement_gain"),
            "| utility=",
            step.get("utility"),
            "| tokens_after=",
            step.get("tokens_after"),
            "| gained_roles=",
            step.get("gained_roles"),
        )


def audit_case(
    *,
    pipeline: C3Pipeline,
    label: str,
    query: str,
) -> None:
    result = pipeline.answer(
        QueryState(
            query=query,
            user_id="user01",
        )
    )

    shadow = result.debug[
        "c3_v3_arbitration_shadow"
    ]

    print()
    print("=" * 110)

    print(
        "CASE:",
        label,
    )

    print(
        "QUERY:",
        query,
    )

    print(
        "QUERY MODE:",
        result.query_mode.value,
    )

    print(
        "COMPARISON STAGE:",
        shadow.get(
            "comparison_stage"
        ),
    )

    print(
        "ACTIVE FOR GENERATION:",
        shadow.get(
            "active_for_generation"
        ),
    )

    print()

    legacy_ids = shadow.get(
        "legacy_selected_ids",
        [],
    )

    c3_ids = shadow.get(
        "c3_v3_selected_ids",
        [],
    )

    print(
        "LEGACY SELECTED:",
        legacy_ids,
    )

    print(
        "C3-V3 SELECTED:",
        c3_ids,
    )

    print(
        "SAME SET:",
        shadow.get(
            "same_set"
        ),
    )

    print(
        "SAME ORDER:",
        shadow.get(
            "same_order"
        ),
    )

    print()

    print(
        "REMOVED BY C3-V3:",
        shadow.get(
            "removed_by_c3_v3",
            [],
        ),
    )

    print(
        "ADDED BY C3-V3:",
        shadow.get(
            "added_by_c3_v3",
            [],
        ),
    )

    print(
        "LEGACY COUNT:",
        shadow.get(
            "legacy_count"
        ),
    )

    print(
        "C3-V3 COUNT:",
        shadow.get(
            "c3_v3_count"
        ),
    )

    print(
        "COUNT DELTA:",
        shadow.get(
            "count_delta"
        ),
    )

    print()

    print(
        "HARD COMPLETE:",
        shadow.get(
            "hard_complete"
        ),
    )

    print(
        "SOFT COMPLETE:",
        shadow.get(
            "soft_complete"
        ),
    )

    print(
        "C3-V3 USED TOKENS:",
        shadow.get(
            "used_tokens"
        ),
    )

    print()

    print(
        "REJECTED INCOMPATIBLE:",
        shadow.get(
            "rejected_incompatible",
            [],
        ),
    )

    print(
        "REJECTED BUDGET:",
        shadow.get(
            "rejected_budget",
            [],
        ),
    )

    print()

    print(
        "ARBITRATION STEPS:"
    )

    print_steps(
        shadow.get(
            "steps",
            [],
        )
    )

    # -----------------------------------------------------
    # High-level diagnostic classification
    # -----------------------------------------------------

    removed = set(
        shadow.get(
            "removed_by_c3_v3",
            [],
        )
    )

    added = set(
        shadow.get(
            "added_by_c3_v3",
            [],
        )
    )

    if (
        not removed
        and not added
    ):
        decision = (
            "UNCHANGED"
        )

    elif (
        removed
        and not added
    ):
        decision = (
            "PRUNED"
        )

    elif (
        added
        and not removed
    ):
        decision = (
            "EXPANDED"
        )

    else:
        decision = (
            "RESTRUCTURED"
        )

    print()

    print(
        "ARBITRATION EFFECT:",
        decision,
    )

    print(
        "FINAL GENERATION SELECTED:",
        result.selected_ids,
    )


def main() -> None:
    pipeline = build_pipeline()

    cases = [
        (
            "CURRENT",
            (
                "What is my current "
                "MSc project scope?"
            ),
        ),
        (
            "HISTORICAL",
            (
                "What was my previous "
                "MSc project scope?"
            ),
        ),
        (
            "TIMELINE",
            (
                "How did my project scope "
                "change over time?"
            ),
        ),
        (
            "PROCEDURAL",
            (
                "When discussing my dissertation "
                "with my supervisor, how should "
                "the answer be written?"
            ),
        ),
        (
            "EXPLANATION",
            (
                "Why did my MSc project "
                "scope change?"
            ),
        ),
    ]

    try:
        for label, query in cases:
            audit_case(
                pipeline=pipeline,
                label=label,
                query=query,
            )

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()