from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.retrievers import (
    InMemoryMemoryStore,
    ProceduralJsonStore,
)
from src.schemas import (
    MemoryType,
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


def test_historical_gap_is_repaired() -> None:
    pipeline = build_pipeline()

    historical_state = QueryState(
        query=(
            "What was my previous "
            "MSc project scope?"
        ),
        user_id="user01",
    )

    try:
        # -------------------------------------------------
        # 1. Compile the historical query normally.
        # -------------------------------------------------

        features = pipeline.analyzer.analyse(
            historical_state.query
        )

        route = pipeline.router.plan(
            features
        )

        compilation = (
            pipeline.requirement_compiler
            .compile_from_legacy(
                query=historical_state.query,
                features=features,
                route=route,
                conflicts=[],
            )
        )

        spec = compilation.spec

        # -------------------------------------------------
        # 2. Deliberately construct an insufficient S0.
        #
        # We intentionally use only the CURRENT memory as
        # the initial selected evidence.
        #
        # This simulates a first-round retrieval/selection
        # miss without corrupting the production pipeline.
        # -------------------------------------------------

        current_state = QueryState(
            query=(
                "What is my current "
                "MSc project scope?"
            ),
            user_id="user01",
        )

        current_features = (
            pipeline.analyzer.analyse(
                current_state.query
            )
        )

        current_candidates = (
            pipeline.memory_store.retrieve(
                memory_type=(
                    MemoryType.SEMANTIC
                ),
                state=current_state,
                features=current_features,
                top_k=20,
                include_archived=True,
            )
        )

        current_candidate = next(
            candidate
            for candidate in current_candidates
            if (
                candidate.memory_id
                == "s_user01_003"
            )
        )

        original_selected = [
            deepcopy(
                current_candidate
            )
        ]

        original_raw_candidates = [
            deepcopy(
                current_candidate
            )
        ]

        # -------------------------------------------------
        # 3. Explicitly represent the first-round gap.
        #
        # answer_target       = satisfied
        # historical_state   = missing
        # -------------------------------------------------

        selector_status = {}

        for requirement in spec.requirements:
            if (
                requirement.role
                == "answer_target"
            ):
                satisfied = (
                    requirement.min_count
                )
            elif (
                requirement.role
                == "historical_state"
            ):
                satisfied = 0
            else:
                # Soft/non-essential roles are left
                # incomplete unless explicitly needed.
                satisfied = 0

            selector_status[
                requirement.role
            ] = {
                "required": (
                    requirement.min_count
                ),
                "satisfied": satisfied,
                "hard": requirement.hard,
                "distinct": (
                    requirement.distinct
                ),
                "complete": (
                    satisfied
                    >= requirement.min_count
                ),
            }

        original_sufficiency = (
            pipeline.requirement_sufficiency
            .evaluate(
                spec=spec,
                selected=original_selected,
                selector_status=(
                    selector_status
                ),
            )
        )

        # -------------------------------------------------
        # 4. Verify S0 is actually insufficient.
        # -------------------------------------------------

        assert (
            not original_sufficiency
            .sufficient
        )

        missing_roles = {
            requirement.role
            for requirement
            in (
                original_sufficiency
                .missing_hard_requirements
            )
        }

        assert (
            "historical_state"
            in missing_roles
        )

        # -------------------------------------------------
        # 5. Create targeted repair plan.
        # -------------------------------------------------

        repair_plan = (
            pipeline.repair_planner.plan(
                spec=spec,
                missing_requirements=(
                    original_sufficiency
                    .missing_hard_requirements
                ),
            )
        )

        assert repair_plan.needed

        assert (
            "historical_state"
            in repair_plan.missing_roles
        )

        assert (
            MemoryType.SEMANTIC
            in repair_plan.target_memory_types
        )

        # -------------------------------------------------
        # 6. Execute ONE bounded repair round.
        # -------------------------------------------------

        execution = (
            pipeline.repair_executor.execute(
                state=historical_state,
                features=features,
                original_route=route,
                spec=spec,
                repair_plan=repair_plan,
                original_raw_candidates=(
                    original_raw_candidates
                ),
                original_selected=(
                    original_selected
                ),
                original_conflicts=[],
                original_sufficiency=(
                    original_sufficiency
                ),
            )
        )

        # -------------------------------------------------
        # 7. Causal assertions:
        #
        # missing requirement
        #       ↓
        # targeted retrieval
        #       ↓
        # historical evidence recovered
        #       ↓
        # hard coverage improves
        #       ↓
        # repair accepted
        # -------------------------------------------------

        assert execution.attempted

        assert execution.accepted

        assert (
            execution.sufficiency
            .hard_requirement_coverage
            >
            original_sufficiency
            .hard_requirement_coverage
        )

        assert (
            execution.sufficiency
            .sufficient
        )

        final_selected_ids = {
            candidate.memory_id
            for candidate
            in execution.selected
        }

        assert (
            "s_user01_004"
            in final_selected_ids
        )

        assert (
            "s_user01_004"
            in execution.repair_retrieved_ids
        )

        assert (
            execution.trace[
                "hard_coverage_after"
            ]
            >
            execution.trace[
                "hard_coverage_before"
            ]
        )

        assert (
            execution.trace[
                "accepted"
            ]
            is True
        )

    finally:
        pipeline.close()