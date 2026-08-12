from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.query_analyzer import QueryAnalyzer
from src.route_planner import RoutePlanner
from src.schemas import (
    MemoryCandidate,
    MemoryType,
)
from src.shared_ranker import SharedRanker


ROOT = Path(__file__).resolve().parents[1]


def make_candidate(
    memory_id: str,
    status: str,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=memory_id,
        memory_type=MemoryType.SEMANTIC,
        text="The MSc project scope is a memory controller.",
        user_id="user01",
        subject="MSc project",
        predicate="scope",
        object_value="memory controller",
        status=status,
        confidence=1.0,
        importance=0.5,
        authority="unknown",
    )


def legacy_score(
    candidate: MemoryCandidate,
    weights: dict,
) -> float:
    score = (
        float(weights["lexical_weight"])
        * candidate.lexical_score

        + float(weights["graph_entity_weight"])
        * candidate.graph_entity_score

        + float(weights["temporal_task_weight"])
        * candidate.temporal_task_score

        + float(weights["validity_weight"])
        * float(
            candidate.metadata[
                "legacy_validity_score"
            ]
        )

        + float(weights["source_confidence_weight"])
        * candidate.source_confidence_score

        + float(weights["route_compatibility_weight"])
        * candidate.route_compatibility_score
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def main() -> None:
    cfg = load_config(
        ROOT
        / "configs"
        / "c3_lite_v2_2_final.yaml"
    )

    analyzer = QueryAnalyzer(cfg)
    router = RoutePlanner(cfg)
    ranker = SharedRanker(cfg)

    query = (
        "What was my previous MSc project scope?"
    )

    features = analyzer.analyse(
        query
    )

    route = router.plan(
        features
    )

    candidates = [
        deepcopy(
            make_candidate(
                "current_state",
                "current",
            )
        ),
        deepcopy(
            make_candidate(
                "historical_state",
                "superseded",
            )
        ),
    ]

    ranker.rank(
        candidates=candidates,
        features=features,
        route=route,
        current_time=datetime.now(
            timezone.utc
        ),
    )

    weights = cfg["ranking"]

    print("=" * 100)
    print(
        "C3-v3 TEMPORAL RANKING CAUSALITY AUDIT"
    )
    print("=" * 100)

    print("QUERY:", query)
    print(
        "QUERY MODE:",
        features.query_mode.value,
    )

    print()
    print(
        "memory_id"
        " | legacy_V"
        " | query_V"
        " | legacy_score"
        " | c3_v3_score"
        " | role"
        " | compatible"
    )

    print("-" * 100)

    rows = []

    for candidate in candidates:
        old_score = legacy_score(
            candidate,
            weights,
        )

        new_score = (
            candidate.final_score
        )

        rows.append(
            (
                candidate.memory_id,
                old_score,
                new_score,
            )
        )

        print(
            candidate.memory_id,
            "|",
            candidate.metadata[
                "legacy_validity_score"
            ],
            "|",
            candidate.validity_score,
            "|",
            round(
                old_score,
                4,
            ),
            "|",
            round(
                new_score,
                4,
            ),
            "|",
            candidate.metadata[
                "query_relative_temporal_role"
            ],
            "|",
            candidate.metadata[
                "query_relative_temporal_compatible"
            ],
        )

    legacy_ranking = sorted(
        rows,
        key=lambda row: row[1],
        reverse=True,
    )

    active_ranking = sorted(
        rows,
        key=lambda row: row[2],
        reverse=True,
    )

    print()
    print(
        "LEGACY RANKING:",
        [
            row[0]
            for row in legacy_ranking
        ],
    )

    print(
        "C3-V3 RANKING:",
        [
            row[0]
            for row in active_ranking
        ],
    )

    inversion = (
        legacy_ranking[0][0]
        != active_ranking[0][0]
    )

    print()
    print(
        "RANK INVERSION:",
        inversion,
    )

    if inversion:
        print(
            "RESULT: query-relative validity "
            "changes the temporal arbitration decision."
        )
    else:
        print(
            "RESULT: no ranking inversion observed."
        )


if __name__ == "__main__":
    main()