from __future__ import annotations

from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.evidence_utility import EvidenceUtilityModel
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import (
    MemoryCandidate,
    QueryState,
)


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


def print_step(
    *,
    step: int,
    candidate: MemoryCandidate,
    utility_result,
    raw_requirement_gain: float,
    selected_before: list[MemoryCandidate],
) -> None:
    print()
    print("-" * 100)

    print(
        f"STEP {step}: ADD {candidate.memory_id}"
    )

    print(
        "SELECTED BEFORE:",
        [
            item.memory_id
            for item in selected_before
        ],
    )

    print(
        "SELECTOR REASON:",
        candidate.metadata.get(
            "selector_reason"
        ),
    )

    print(
        "EVIDENCE ROLES:",
        candidate.metadata.get(
            "evidence_roles",
            [],
        ),
    )

    print(
        "RAW SELECTOR REQUIREMENT GAIN:",
        raw_requirement_gain,
    )

    print()

    print(
        "relevance                =",
        utility_result.relevance,
    )

    print(
        "temporal_validity        =",
        utility_result.temporal_validity,
    )

    print(
        "provenance               =",
        utility_result.provenance,
    )

    print(
        "requirement_gain         =",
        utility_result.requirement_gain,
    )

    print(
        "coverage_gain            =",
        utility_result.coverage_gain,
    )

    print(
        "incompatibility          =",
        utility_result.incompatibility,
    )

    print(
        "redundancy               =",
        utility_result.redundancy,
    )

    print(
        "token_cost               =",
        utility_result.token_cost,
    )

    print(
        "normalised_token_cost    =",
        utility_result.normalised_token_cost,
    )

    print()

    print(
        "MARGINAL UTILITY ΔU      =",
        utility_result.utility,
    )


def audit_query(
    *,
    pipeline: C3Pipeline,
    utility_model: EvidenceUtilityModel,
    label: str,
    query: str,
) -> None:
    result = pipeline.answer(
        QueryState(
            query=query,
            user_id="user01",
        )
    )

    features = (
        pipeline.analyzer.analyse(
            query
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
        "FINAL COVERAGE:",
        result.coverage,
    )

    selected_so_far: list[
        MemoryCandidate
    ] = []

    utilities: list[
        tuple[str, float]
    ] = []

    for step, candidate in enumerate(
        result.selected_evidence,
        start=1,
    ):
        raw_requirement_gain = float(
            candidate.metadata.get(
                "selector_requirement_gain",
                0.0,
            )
            or 0.0
        )

        utility_result = (
            utility_model.evaluate(
                candidate=candidate,
                selected=selected_so_far,
                features=features,
                requirement_gain=(
                    raw_requirement_gain
                ),
            )
        )

        print_step(
            step=step,
            candidate=candidate,
            utility_result=utility_result,
            raw_requirement_gain=(
                raw_requirement_gain
            ),
            selected_before=(
                selected_so_far
            ),
        )

        utilities.append(
            (
                candidate.memory_id,
                utility_result.utility,
            )
        )

        selected_so_far.append(
            candidate
        )

    print()
    print("-" * 100)

    print(
        "SEQUENTIAL UTILITIES:"
    )

    for memory_id, utility in utilities:
        print(
            memory_id,
            "=>",
            utility,
        )

    if len(utilities) >= 2:
        first_id, first_utility = (
            utilities[0]
        )

        print()
        print(
            "UTILITY AFTER FIRST EVIDENCE:"
        )

        for memory_id, utility in (
            utilities[1:]
        ):
            print(
                memory_id,
                "relative_to_first=",
                round(
                    utility
                    - first_utility,
                    6,
                ),
            )


def main() -> None:
    pipeline = build_pipeline()

    utility_model = (
        EvidenceUtilityModel(
            pipeline.config,
            pipeline.coverage,
        )
    )

    cases = [
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
                "Why did my MSc project scope "
                "change?"
            ),
        ),
    ]

    try:
        for label, query in cases:
            audit_query(
                pipeline=pipeline,
                utility_model=utility_model,
                label=label,
                query=query,
            )

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()