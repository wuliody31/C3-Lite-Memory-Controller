from __future__ import annotations

from pathlib import Path
from typing import Any

from src.backbones import MockBackbone
from src.config import load_config
from src.evidence_requirements import (
    EvidenceRequirement,
)
from src.evidence_utility import (
    EvidenceUtilityModel,
)
from src.pipeline import C3Pipeline
from src.requirement_gain_v3 import (
    QueryConsistentRequirementGain,
)
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


def extract_requirements(
    result: Any,
) -> list[EvidenceRequirement]:
    """Reconstruct compiled EvidenceRequirement objects from debug trace."""

    spec = result.debug[
        "c3_v3_requirement_spec"
    ]

    raw_requirements = spec.get(
        "requirements",
        [],
    )

    requirements: list[
        EvidenceRequirement
    ] = []

    for item in raw_requirements:
        if isinstance(
            item,
            EvidenceRequirement,
        ):
            requirements.append(
                item
            )
            continue

        if not isinstance(
            item,
            dict,
        ):
            continue

        requirements.append(
            EvidenceRequirement(
                role=str(
                    item.get(
                        "role",
                        "",
                    )
                ),
                min_count=int(
                    item.get(
                        "min_count",
                        1,
                    )
                ),
                hard=bool(
                    item.get(
                        "hard",
                        True,
                    )
                ),
                distinct=bool(
                    item.get(
                        "distinct",
                        False,
                    )
                ),
                description=str(
                    item.get(
                        "description",
                        "",
                    )
                    or ""
                ),
            )
        )

    return [
        requirement
        for requirement
        in requirements
        if requirement.role
    ]


def print_requirements(
    requirements: list[
        EvidenceRequirement
    ],
) -> None:
    print(
        "COMPILED REQUIREMENTS:"
    )

    if not requirements:
        print(
            "  <none>"
        )
        return

    for requirement in requirements:
        print(
            " ",
            requirement.role,
            "| hard=",
            requirement.hard,
            "| min_count=",
            requirement.min_count,
            "| distinct=",
            requirement.distinct,
        )


def print_step(
    *,
    step: int,
    candidate: MemoryCandidate,
    selected_before: list[
        MemoryCandidate
    ],
    legacy_requirement_gain: float,
    consistent_gain,
    legacy_utility,
    consistent_utility,
) -> None:
    print()
    print(
        "-" * 100
    )

    print(
        f"STEP {step}: ADD "
        f"{candidate.memory_id}"
    )

    print(
        "SELECTED BEFORE:",
        [
            item.memory_id
            for item
            in selected_before
        ],
    )

    print(
        "SELECTOR REASON:",
        candidate.metadata.get(
            "selector_reason"
        ),
    )

    print(
        "LEGACY EVIDENCE ROLES:",
        candidate.metadata.get(
            "evidence_roles",
            [],
        ),
    )

    print(
        "QUERY TEMPORAL ROLE:",
        candidate.metadata.get(
            "query_relative_temporal_role"
        ),
    )

    print(
        "TEMPORAL COMPATIBLE:",
        candidate.metadata.get(
            "query_relative_temporal_compatible"
        ),
    )

    print()

    print(
        "LEGACY SELECTOR REQUIREMENT GAIN:",
        legacy_requirement_gain,
    )

    print(
        "C3-V3 EFFECTIVE ROLES:",
        consistent_gain.effective_roles,
    )

    print(
        "C3-V3 GAINED ROLES:",
        consistent_gain.gained_roles,
    )

    print(
        "C3-V3 REQUIREMENT COVERAGE BEFORE:",
        consistent_gain.before_coverage,
    )

    print(
        "C3-V3 REQUIREMENT COVERAGE AFTER:",
        consistent_gain.after_coverage,
    )

    print(
        "C3-V3 REQUIREMENT GAIN:",
        consistent_gain.gain,
    )

    print(
        "C3-V3 HARD GAIN:",
        consistent_gain.hard_gain,
    )

    print(
        "C3-V3 SOFT GAIN:",
        consistent_gain.soft_gain,
    )

    print()

    print(
        "relevance             =",
        consistent_utility.relevance,
    )

    print(
        "temporal_validity     =",
        consistent_utility.temporal_validity,
    )

    print(
        "provenance            =",
        consistent_utility.provenance,
    )

    print(
        "coverage_gain         =",
        consistent_utility.coverage_gain,
    )

    print(
        "incompatibility       =",
        consistent_utility.incompatibility,
    )

    print(
        "redundancy            =",
        consistent_utility.redundancy,
    )

    print(
        "token_cost            =",
        consistent_utility.token_cost,
    )

    print(
        "normalised_token_cost =",
        consistent_utility.normalised_token_cost,
    )

    print()

    print(
        "LEGACY-GAIN UTILITY   =",
        legacy_utility.utility,
    )

    print(
        "C3-V3 UTILITY ΔU      =",
        consistent_utility.utility,
    )

    print(
        "UTILITY CHANGE        =",
        round(
            consistent_utility.utility
            - legacy_utility.utility,
            6,
        ),
    )


def audit_query(
    *,
    pipeline: C3Pipeline,
    utility_model: EvidenceUtilityModel,
    requirement_gain_model: (
        QueryConsistentRequirementGain
    ),
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

    requirements = (
        extract_requirements(
            result
        )
    )

    print()
    print(
        "=" * 100
    )

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
        "FINAL SELECTED:",
        result.selected_ids,
    )

    print(
        "FINAL COVERAGE:",
        result.coverage,
    )

    print()

    print_requirements(
        requirements
    )

    selected_so_far: list[
        MemoryCandidate
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    for step, candidate in enumerate(
        result.selected_evidence,
        start=1,
    ):
        legacy_requirement_gain = float(
            candidate.metadata.get(
                "selector_requirement_gain",
                0.0,
            )
            or 0.0
        )

        consistent_gain = (
            requirement_gain_model.evaluate(
                candidate=candidate,
                selected=selected_so_far,
                requirements=requirements,
            )
        )

        legacy_utility = (
            utility_model.evaluate(
                candidate=candidate,
                selected=selected_so_far,
                features=features,
                requirement_gain=(
                    legacy_requirement_gain
                ),
            )
        )

        consistent_utility = (
            utility_model.evaluate(
                candidate=candidate,
                selected=selected_so_far,
                features=features,
                requirement_gain=(
                    consistent_gain.gain
                ),
            )
        )

        print_step(
            step=step,
            candidate=candidate,
            selected_before=(
                selected_so_far
            ),
            legacy_requirement_gain=(
                legacy_requirement_gain
            ),
            consistent_gain=(
                consistent_gain
            ),
            legacy_utility=(
                legacy_utility
            ),
            consistent_utility=(
                consistent_utility
            ),
        )

        summary_rows.append(
            {
                "memory_id": (
                    candidate.memory_id
                ),
                "legacy_gain": (
                    legacy_requirement_gain
                ),
                "consistent_gain": (
                    consistent_gain.gain
                ),
                "hard_gain": (
                    consistent_gain.hard_gain
                ),
                "soft_gain": (
                    consistent_gain.soft_gain
                ),
                "legacy_utility": (
                    legacy_utility.utility
                ),
                "utility": (
                    consistent_utility.utility
                ),
                "compatible": (
                    candidate.metadata.get(
                        "query_relative_temporal_compatible"
                    )
                ),
            }
        )

        selected_so_far.append(
            candidate
        )

    print()
    print(
        "-" * 100
    )

    print(
        "FINAL STEP SUMMARY:"
    )

    for row in summary_rows:
        print(
            row[
                "memory_id"
            ],
            "| legacy_gain=",
            row[
                "legacy_gain"
            ],
            "| valid_gain=",
            row[
                "consistent_gain"
            ],
            "| hard_gain=",
            row[
                "hard_gain"
            ],
            "| soft_gain=",
            row[
                "soft_gain"
            ],
            "| legacy_U=",
            row[
                "legacy_utility"
            ],
            "| c3_U=",
            row[
                "utility"
            ],
            "| compatible=",
            row[
                "compatible"
            ],
        )


def main() -> None:
    pipeline = build_pipeline()

    utility_model = (
        EvidenceUtilityModel(
            pipeline.config,
            pipeline.coverage,
        )
    )

    requirement_gain_model = (
        QueryConsistentRequirementGain()
    )

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
            audit_query(
                pipeline=pipeline,
                utility_model=utility_model,
                requirement_gain_model=(
                    requirement_gain_model
                ),
                label=label,
                query=query,
            )

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()