from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


# =========================================================
# Generic I/O helpers
# =========================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            try:
                rows.append(
                    json.loads(
                        text
                    )
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

    return rows


def read_csv_index(
    path: Path,
    *,
    key_field: str,
) -> dict[str, dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    output: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        key = str(
            row.get(
                key_field,
                "",
            )
        ).strip()

        if not key:
            raise ValueError(
                f"{path}: missing {key_field}"
            )

        if key in output:
            raise ValueError(
                f"{path}: duplicate {key_field}={key}"
            )

        output[
            key
        ] = dict(
            row
        )

    return output


def write_csv(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows to write: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =========================================================
# Numeric / boolean helpers
# =========================================================


def as_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def round_optional(
    value: float | None,
    digits: int = 6,
) -> float | None:
    if value is None:
        return None

    return round(
        float(
            value
        ),
        digits,
    )


# =========================================================
# Exact 2x2 statistics
# =========================================================


def fisher_two_sided(
    a: int,
    b: int,
    c: int,
    d: int,
) -> float:
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def probability(
        x: int,
    ) -> float:
        y = row1 - x
        z = col1 - x
        w = row2 - z

        if min(
            x,
            y,
            z,
            w,
        ) < 0:
            return 0.0

        return (
            math.comb(
                col1,
                x,
            )
            * math.comb(
                total - col1,
                row1 - x,
            )
            / math.comb(
                total,
                row1,
            )
        )

    low = max(
        0,
        row1
        - (
            total
            - col1
        ),
    )

    high = min(
        row1,
        col1,
    )

    observed = probability(
        a
    )

    p_value = 0.0

    for x in range(
        low,
        high + 1,
    ):
        p = probability(
            x
        )

        if (
            p
            <= observed
            + 1e-15
        ):
            p_value += p

    return min(
        1.0,
        p_value,
    )


def binary_stats(
    cases: list[
        dict[str, Any]
    ],
    *,
    target: str,
    predictor: str,
) -> dict[str, Any]:
    positive = [
        case
        for case in cases
        if as_bool(
            case[
                target
            ]
        )
    ]

    negative = [
        case
        for case in cases
        if not as_bool(
            case[
                target
            ]
        )
    ]

    a = sum(
        as_bool(
            case[
                predictor
            ]
        )
        for case
        in positive
    )

    b = sum(
        as_bool(
            case[
                predictor
            ]
        )
        for case
        in negative
    )

    c = len(
        positive
    ) - a

    d = len(
        negative
    ) - b

    precision = (
        a
        / (
            a + b
        )
        if (
            a + b
        )
        else 0.0
    )

    recall = (
        a
        / len(
            positive
        )
        if positive
        else 0.0
    )

    false_positive_rate = (
        b
        / len(
            negative
        )
        if negative
        else 0.0
    )

    specificity = (
        d
        / len(
            negative
        )
        if negative
        else 0.0
    )

    odds_ratio = (
        (
            a
            + 0.5
        )
        * (
            d
            + 0.5
        )
        / (
            (
                b
                + 0.5
            )
            * (
                c
                + 0.5
            )
        )
    )

    return {
        "target": target,
        "predictor": predictor,
        "harm_flagged": a,
        "harm_not_flagged": c,
        "no_harm_flagged": b,
        "no_harm_not_flagged": d,
        "precision_for_harm": round(
            precision,
            6,
        ),
        "recall_for_harm": round(
            recall,
            6,
        ),
        "false_positive_rate": round(
            false_positive_rate,
            6,
        ),
        "specificity": round(
            specificity,
            6,
        ),
        "odds_ratio_corrected": round(
            odds_ratio,
            6,
        ),
        "fisher_two_sided_p": round(
            fisher_two_sided(
                a,
                b,
                c,
                d,
            ),
            6,
        ),
    }


# =========================================================
# Slot-support preservation helpers
# =========================================================


ANSWER_CRITICAL_ROLES = {
    "answer_target",
    "current_state",
    "historical_state",
    "preferred_resolution",
    "transition",
    "supporting_evidence",
}


def status_index(
    statuses: Any,
) -> dict[
    str,
    dict[str, Any],
]:
    output: dict[
        str,
        dict[str, Any],
    ] = {}

    if not isinstance(
        statuses,
        list,
    ):
        return output

    for status in statuses:
        if not isinstance(
            status,
            dict,
        ):
            continue

        slot_id = str(
            status.get(
                "slot_id",
                "",
            )
        ).strip()

        if not slot_id:
            continue

        output[
            slot_id
        ] = status

    return output


def support_ids(
    status: dict[
        str,
        Any,
    ]
    | None,
) -> list[str]:
    if status is None:
        return []

    raw = status.get(
        "supporting_memory_ids",
        [],
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    return [
        str(
            item
        )
        for item in raw
    ]


def evidence_roles(
    trace: dict[
        str,
        Any,
    ]
    | None,
) -> set[str]:
    if trace is None:
        return set()

    raw = trace.get(
        "evidence_roles",
        [],
    )

    if isinstance(
        raw,
        str,
    ):
        return {
            raw
        }

    if isinstance(
        raw,
        (
            list,
            tuple,
            set,
        ),
    ):
        return {
            str(
                item
            )
            for item in raw
        }

    return set()


def trace_index(
    prediction: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    trace_rows = (
        prediction
        .get(
            "debug",
            {},
        )
        .get(
            "full_candidate_score_trace",
            [],
        )
    )

    output: dict[
        str,
        dict[str, Any],
    ] = {}

    if not isinstance(
        trace_rows,
        list,
    ):
        return output

    for row in trace_rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        memory_id = str(
            row.get(
                "memory_id",
                "",
            )
        ).strip()

        if memory_id:
            output[
                memory_id
            ] = row

    return output


def weighted_retention(
    *,
    legacy_ids: set[str],
    retained_ids: set[str],
    traces: dict[
        str,
        dict[str, Any],
    ],
    field: str,
) -> float | None:
    weights: dict[
        str,
        float,
    ] = {
        memory_id: max(
            0.0,
            safe_float(
                traces
                .get(
                    memory_id,
                    {},
                )
                .get(
                    field
                ),
                0.0,
            ),
        )
        for memory_id
        in legacy_ids
    }

    denominator = sum(
        weights.values()
    )

    if denominator <= 0.0:
        return None

    numerator = sum(
        weights[
            memory_id
        ]
        for memory_id
        in retained_ids
        if memory_id
        in weights
    )

    return (
        numerator
        / denominator
    )


def slot_preservation_rows(
    *,
    question_id: str,
    shadow: dict[
        str,
        Any,
    ],
    traces: dict[
        str,
        dict[str, Any],
    ],
) -> list[
    dict[str, Any]
]:
    legacy_statuses = (
        status_index(
            shadow.get(
                "legacy_slot_statuses",
                [],
            )
        )
    )

    c3_statuses = (
        status_index(
            shadow.get(
                "c3_v3_slot_statuses",
                [],
            )
        )
    )

    slot_ids = sorted(
        set(
            legacy_statuses
        )
        | set(
            c3_statuses
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for slot_id in slot_ids:
        legacy = (
            legacy_statuses
            .get(
                slot_id
            )
        )

        c3 = (
            c3_statuses
            .get(
                slot_id
            )
        )

        reference = (
            legacy
            or c3
            or {}
        )

        legacy_ids = set(
            support_ids(
                legacy
            )
        )

        c3_ids = set(
            support_ids(
                c3
            )
        )

        retained = (
            legacy_ids
            & c3_ids
        )

        removed = (
            legacy_ids
            - c3_ids
        )

        added = (
            c3_ids
            - legacy_ids
        )

        legacy_complete = (
            as_bool(
                legacy.get(
                    "complete",
                    False,
                )
            )
            if legacy
            else False
        )

        c3_complete = (
            as_bool(
                c3.get(
                    "complete",
                    False,
                )
            )
            if c3
            else False
        )

        unweighted_retention = (
            len(
                retained
            )
            / len(
                legacy_ids
            )
            if legacy_ids
            else None
        )

        requirement_gain_retention = (
            weighted_retention(
                legacy_ids=legacy_ids,
                retained_ids=retained,
                traces=traces,
                field=(
                    "selector_requirement_gain"
                ),
            )
        )

        topical_retention = (
            weighted_retention(
                legacy_ids=legacy_ids,
                retained_ids=retained,
                traces=traces,
                field=(
                    "selector_topical_score"
                ),
            )
        )

        full_replacement = (
            legacy_complete
            and c3_complete
            and bool(
                legacy_ids
            )
            and bool(
                c3_ids
            )
            and not bool(
                retained
            )
        )

        support_count_drop_complete = (
            legacy_complete
            and c3_complete
            and (
                len(
                    c3_ids
                )
                < len(
                    legacy_ids
                )
            )
        )

        rows.append(
            {
                "question_id": (
                    question_id
                ),
                "slot_id": (
                    slot_id
                ),
                "kind": str(
                    reference.get(
                        "kind",
                        "",
                    )
                ),
                "target": str(
                    reference.get(
                        "target",
                        "",
                    )
                ),
                "hard": (
                    as_bool(
                        reference.get(
                            "hard",
                            False,
                        )
                    )
                ),
                "legacy_complete": (
                    legacy_complete
                ),
                "c3_v3_complete": (
                    c3_complete
                ),
                "legacy_support_count": (
                    len(
                        legacy_ids
                    )
                ),
                "c3_v3_support_count": (
                    len(
                        c3_ids
                    )
                ),
                "support_count_delta": (
                    len(
                        c3_ids
                    )
                    - len(
                        legacy_ids
                    )
                ),
                "retained_support_count": (
                    len(
                        retained
                    )
                ),
                "legacy_support_ids": (
                    ";".join(
                        sorted(
                            legacy_ids
                        )
                    )
                ),
                "c3_v3_support_ids": (
                    ";".join(
                        sorted(
                            c3_ids
                        )
                    )
                ),
                "retained_support_ids": (
                    ";".join(
                        sorted(
                            retained
                        )
                    )
                ),
                "removed_support_ids": (
                    ";".join(
                        sorted(
                            removed
                        )
                    )
                ),
                "added_support_ids": (
                    ";".join(
                        sorted(
                            added
                        )
                    )
                ),
                "slot_support_retention": (
                    round_optional(
                        unweighted_retention
                    )
                ),
                "requirement_gain_weighted_retention": (
                    round_optional(
                        requirement_gain_retention
                    )
                ),
                "topical_score_weighted_retention": (
                    round_optional(
                        topical_retention
                    )
                ),
                "full_slot_replacement": (
                    full_replacement
                ),
                "support_count_drop_while_complete": (
                    support_count_drop_complete
                ),
            }
        )

    return rows


def minimum_available(
    values: list[
        float | None
    ],
) -> float | None:
    available = [
        value
        for value in values
        if value is not None
    ]

    if not available:
        return None

    return min(
        available
    )


def mean_available(
    values: list[
        float | None
    ],
) -> float | None:
    available = [
        value
        for value in values
        if value is not None
    ]

    if not available:
        return None

    return (
        sum(
            available
        )
        / len(
            available
        )
    )


# =========================================================
# Main
# =========================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "M2-C3.6 diagnostic audit: extend paired semantic-slot "
            "deficit analysis with slot-support preservation features."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--semantic-labels",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_rows = (
        read_jsonl(
            args.predictions
        )
    )

    if len(
        prediction_rows
    ) != 60:
        raise AssertionError(
            "Expected 60 Full60 predictions, "
            f"got {len(prediction_rows)}"
        )

    predictions = {
        str(
            row[
                "question_id"
            ]
        ): row
        for row
        in prediction_rows
    }

    labels = read_csv_index(
        args.semantic_labels,
        key_field="question_id",
    )

    if len(
        labels
    ) != 22:
        raise AssertionError(
            "Expected 22 semantic labels from the removed-gold audit, "
            f"got {len(labels)}"
        )

    missing_predictions = (
        set(
            labels
        )
        - set(
            predictions
        )
    )

    if missing_predictions:
        raise AssertionError(
            "Semantic labels missing from predictions: "
            + ", ".join(
                sorted(
                    missing_predictions
                )
            )
        )

    cases: list[
        dict[str, Any]
    ] = []

    slot_rows: list[
        dict[str, Any]
    ] = []

    for question_id in sorted(
        labels
    ):
        prediction = (
            predictions[
                question_id
            ]
        )

        shadow = (
            prediction
            .get(
                "debug",
                {},
            )
            .get(
                "c3_v3_arbitration_shadow",
                {},
            )
        )

        if not shadow.get(
            "available",
            False,
        ):
            raise AssertionError(
                f"{question_id}: arbitration shadow unavailable"
            )

        required_fields = (
            "legacy_slot_sufficient",
            "c3_v3_slot_sufficient",
            "legacy_hard_slot_coverage",
            "c3_v3_hard_slot_coverage",
            "introduced_slot_deficit",
            "lost_slots",
            "legacy_slot_statuses",
            "c3_v3_slot_statuses",
            "removed_by_c3_v3",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in shadow
        ]

        if missing_fields:
            raise AssertionError(
                f"{question_id}: missing M2-C3.4 fields: "
                + ", ".join(
                    missing_fields
                )
            )

        label = labels[
            question_id
        ]

        legacy_slot_sufficient = (
            as_bool(
                shadow[
                    "legacy_slot_sufficient"
                ]
            )
        )

        c3_v3_slot_sufficient = (
            as_bool(
                shadow[
                    "c3_v3_slot_sufficient"
                ]
            )
        )

        introduced_slot_deficit = (
            as_bool(
                shadow[
                    "introduced_slot_deficit"
                ]
            )
        )

        lost_slots = list(
            shadow.get(
                "lost_slots",
                [],
            )
        )

        legacy_coverage = float(
            shadow[
                "legacy_hard_slot_coverage"
            ]
        )

        c3_coverage = float(
            shadow[
                "c3_v3_hard_slot_coverage"
            ]
        )

        slot_coverage_delta = float(
            shadow.get(
                "slot_coverage_delta",
                (
                    c3_coverage
                    - legacy_coverage
                ),
            )
        )

        traces = (
            trace_index(
                prediction
            )
        )

        question_slot_rows = (
            slot_preservation_rows(
                question_id=(
                    question_id
                ),
                shadow=shadow,
                traces=traces,
            )
        )

        slot_rows.extend(
            question_slot_rows
        )

        # For preservation diagnostics, focus on hard slots that were
        # complete in Legacy and remain complete in C3. These are exactly
        # the cases where binary occupancy can hide substitution/collapse.
        complete_to_complete = [
            row
            for row in question_slot_rows
            if (
                as_bool(
                    row[
                        "hard"
                    ]
                )
                and as_bool(
                    row[
                        "legacy_complete"
                    ]
                )
                and as_bool(
                    row[
                        "c3_v3_complete"
                    ]
                )
            )
        ]

        retention_values = [
            (
                safe_float(
                    row[
                        "slot_support_retention"
                    ]
                )
                if (
                    row[
                        "slot_support_retention"
                    ]
                    is not None
                )
                else None
            )
            for row in complete_to_complete
        ]

        requirement_weighted_values = [
            (
                safe_float(
                    row[
                        "requirement_gain_weighted_retention"
                    ]
                )
                if (
                    row[
                        "requirement_gain_weighted_retention"
                    ]
                    is not None
                )
                else None
            )
            for row in complete_to_complete
        ]

        topical_weighted_values = [
            (
                safe_float(
                    row[
                        "topical_score_weighted_retention"
                    ]
                )
                if (
                    row[
                        "topical_score_weighted_retention"
                    ]
                    is not None
                )
                else None
            )
            for row in complete_to_complete
        ]

        min_retention = (
            minimum_available(
                retention_values
            )
        )

        mean_retention = (
            mean_available(
                retention_values
            )
        )

        min_requirement_weighted = (
            minimum_available(
                requirement_weighted_values
            )
        )

        min_topical_weighted = (
            minimum_available(
                topical_weighted_values
            )
        )

        any_full_slot_replacement = (
            any(
                as_bool(
                    row[
                        "full_slot_replacement"
                    ]
                )
                for row
                in complete_to_complete
            )
        )

        any_support_count_drop = (
            any(
                as_bool(
                    row[
                        "support_count_drop_while_complete"
                    ]
                )
                for row
                in complete_to_complete
            )
        )

        zero_support_retention = (
            min_retention is not None
            and min_retention <= 0.0
        )

        support_retention_le_half = (
            min_retention is not None
            and min_retention <= 0.5
        )

        support_retention_lt_one = (
            min_retention is not None
            and min_retention < 1.0
        )

        legacy_hard_support_union: set[
            str
        ] = set()

        c3_hard_support_union: set[
            str
        ] = set()

        for row in complete_to_complete:
            legacy_hard_support_union.update(
                item
                for item
                in str(
                    row[
                        "legacy_support_ids"
                    ]
                ).split(
                    ";"
                )
                if item
            )

            c3_hard_support_union.update(
                item
                for item
                in str(
                    row[
                        "c3_v3_support_ids"
                    ]
                ).split(
                    ";"
                )
                if item
            )

        single_candidate_collapse = (
            legacy_slot_sufficient
            and c3_v3_slot_sufficient
            and len(
                legacy_hard_support_union
            )
            >= 2
            and len(
                c3_hard_support_union
            )
            == 1
        )

        removed_ids = [
            str(
                item
            )
            for item
            in shadow.get(
                "removed_by_c3_v3",
                [],
            )
        ]

        removed_anchor_ids: list[
            str
        ] = []

        removed_answer_target_ids: list[
            str
        ] = []

        removed_transition_ids: list[
            str
        ] = []

        removed_support_ids: list[
            str
        ] = []

        removed_current_state_ids: list[
            str
        ] = []

        for memory_id in removed_ids:
            roles = evidence_roles(
                traces.get(
                    memory_id
                )
            )

            if (
                roles
                & ANSWER_CRITICAL_ROLES
            ):
                removed_anchor_ids.append(
                    memory_id
                )

            if (
                "answer_target"
                in roles
            ):
                removed_answer_target_ids.append(
                    memory_id
                )

            if (
                "transition"
                in roles
            ):
                removed_transition_ids.append(
                    memory_id
                )

            if (
                "supporting_evidence"
                in roles
            ):
                removed_support_ids.append(
                    memory_id
                )

            if (
                "current_state"
                in roles
            ):
                removed_current_state_ids.append(
                    memory_id
                )

        answer_anchor_removed = bool(
            removed_anchor_ids
        )

        # Natural combined diagnostic:
        # occupancy loss OR within-slot support preservation loss.
        occupancy_or_support_loss = (
            introduced_slot_deficit
            or any_full_slot_replacement
            or any_support_count_drop
        )

        cases.append(
            {
                "question_id": question_id,
                "semantic_label": (
                    label.get(
                        "semantic_label",
                        "",
                    )
                ),
                "strict_semantic_harm": (
                    label.get(
                        "strict_semantic_harm",
                        "0",
                    )
                ),
                "broad_semantic_loss": (
                    label.get(
                        "broad_semantic_loss",
                        "0",
                    )
                ),

                # Original M2-C3.4 predictors.
                "legacy_slot_sufficient": (
                    legacy_slot_sufficient
                ),
                "c3_v3_slot_sufficient": (
                    c3_v3_slot_sufficient
                ),
                "introduced_slot_deficit": (
                    introduced_slot_deficit
                ),
                "c3_slot_insufficient": (
                    not c3_v3_slot_sufficient
                ),
                "any_lost_slot": bool(
                    lost_slots
                ),
                "slot_coverage_drop": (
                    slot_coverage_delta
                    < 0.0
                ),

                # M2-C3.6 support-preservation diagnostics.
                "any_full_slot_replacement": (
                    any_full_slot_replacement
                ),
                "any_support_count_drop_complete": (
                    any_support_count_drop
                ),
                "zero_support_retention_complete": (
                    zero_support_retention
                ),
                "support_retention_le_half_complete": (
                    support_retention_le_half
                ),
                "support_retention_lt_one_complete": (
                    support_retention_lt_one
                ),
                "single_candidate_collapse": (
                    single_candidate_collapse
                ),
                "answer_anchor_removed": (
                    answer_anchor_removed
                ),
                "answer_target_removed": bool(
                    removed_answer_target_ids
                ),
                "transition_removed": bool(
                    removed_transition_ids
                ),
                "supporting_evidence_removed": bool(
                    removed_support_ids
                ),
                "current_state_removed": bool(
                    removed_current_state_ids
                ),
                "occupancy_or_support_loss": (
                    occupancy_or_support_loss
                ),

                # Continuous descriptive diagnostics.
                "min_complete_slot_support_retention": (
                    round_optional(
                        min_retention
                    )
                ),
                "mean_complete_slot_support_retention": (
                    round_optional(
                        mean_retention
                    )
                ),
                "min_requirement_gain_weighted_retention": (
                    round_optional(
                        min_requirement_weighted
                    )
                ),
                "min_topical_score_weighted_retention": (
                    round_optional(
                        min_topical_weighted
                    )
                ),

                "legacy_complete_support_union_count": (
                    len(
                        legacy_hard_support_union
                    )
                ),
                "c3_complete_support_union_count": (
                    len(
                        c3_hard_support_union
                    )
                ),

                # Existing audit fields.
                "legacy_hard_slot_coverage": (
                    legacy_coverage
                ),
                "c3_v3_hard_slot_coverage": (
                    c3_coverage
                ),
                "slot_coverage_delta": (
                    slot_coverage_delta
                ),
                "lost_slots": ";".join(
                    lost_slots
                ),
                "gained_slots": ";".join(
                    shadow.get(
                        "gained_slots",
                        [],
                    )
                ),
                "removed_by_c3_v3": ";".join(
                    removed_ids
                ),
                "removed_answer_anchor_ids": ";".join(
                    removed_anchor_ids
                ),
                "removed_answer_target_ids": ";".join(
                    removed_answer_target_ids
                ),
                "removed_transition_ids": ";".join(
                    removed_transition_ids
                ),
                "removed_supporting_evidence_ids": ";".join(
                    removed_support_ids
                ),
                "removed_current_state_ids": ";".join(
                    removed_current_state_ids
                ),
                "legacy_selected_ids": ";".join(
                    shadow.get(
                        "legacy_selected_ids",
                        [],
                    )
                ),
                "c3_v3_selected_ids": ";".join(
                    shadow.get(
                        "c3_v3_selected_ids",
                        [],
                    )
                ),
            }
        )

    # =====================================================
    # Diagnostic statistics
    #
    # IMPORTANT:
    # These 22 labels have already been inspected during development.
    # The M2-C3.6 results are diagnostic, not unbiased final validation.
    # =====================================================

    targets = (
        "strict_semantic_harm",
        "broad_semantic_loss",
    )

    predictors = (
        # Original frozen M2-C3.4 reference predictors.
        "introduced_slot_deficit",
        "c3_slot_insufficient",

        # New M2-C3.6 preservation predictors.
        "any_full_slot_replacement",
        "any_support_count_drop_complete",
        "zero_support_retention_complete",
        "support_retention_le_half_complete",
        "support_retention_lt_one_complete",
        "single_candidate_collapse",
        "answer_anchor_removed",
        "answer_target_removed",
        "transition_removed",
        "supporting_evidence_removed",
        "current_state_removed",
        "occupancy_or_support_loss",
    )

    stats = [
        binary_stats(
            cases,
            target=target,
            predictor=predictor,
        )
        for target
        in targets
        for predictor
        in predictors
    ]

    strict_reference = next(
        row
        for row
        in stats
        if (
            row[
                "target"
            ]
            == "strict_semantic_harm"
            and row[
                "predictor"
            ]
            == "introduced_slot_deficit"
        )
    )

    broad_reference = next(
        row
        for row
        in stats
        if (
            row[
                "target"
            ]
            == "broad_semantic_loss"
            and row[
                "predictor"
            ]
            == "introduced_slot_deficit"
        )
    )

    slot_transition_counts = (
        Counter()
    )

    for case in cases:
        pair = (
            bool(
                case[
                    "legacy_slot_sufficient"
                ]
            ),
            bool(
                case[
                    "c3_v3_slot_sufficient"
                ]
            ),
        )

        labels_by_pair = {
            (
                True,
                True,
            ): "legacy_true_c3_true",
            (
                True,
                False,
            ): "legacy_true_c3_false",
            (
                False,
                True,
            ): "legacy_false_c3_true",
            (
                False,
                False,
            ): "legacy_false_c3_false",
        }

        slot_transition_counts[
            labels_by_pair[
                pair
            ]
        ] += 1

    strict_harm_cases = [
        case
        for case
        in cases
        if as_bool(
            case[
                "strict_semantic_harm"
            ]
        )
    ]

    strict_baseline_incomplete = [
        case[
            "question_id"
        ]
        for case
        in strict_harm_cases
        if not as_bool(
            case[
                "legacy_slot_sufficient"
            ]
        )
    ]

    strict_complete_to_complete_harm = [
        case[
            "question_id"
        ]
        for case
        in strict_harm_cases
        if (
            as_bool(
                case[
                    "legacy_slot_sufficient"
                ]
            )
            and as_bool(
                case[
                    "c3_v3_slot_sufficient"
                ]
            )
        )
    ]

    summary = {
        "stage": (
            "M2-C3.6 Slot Support Preservation Audit"
        ),
        "questions": len(
            cases
        ),
        "strict_semantic_harm_questions": sum(
            as_bool(
                case[
                    "strict_semantic_harm"
                ]
            )
            for case
            in cases
        ),
        "broad_semantic_loss_questions": sum(
            as_bool(
                case[
                    "broad_semantic_loss"
                ]
            )
            for case
            in cases
        ),
        "slot_sufficiency_transitions": dict(
            slot_transition_counts
        ),
        "m2_c3_4_reference": {
            "strict_semantic_harm": (
                strict_reference
            ),
            "broad_semantic_loss": (
                broad_reference
            ),
        },
        "strict_harm_failure_decomposition": {
            "baseline_incomplete_question_ids": (
                strict_baseline_incomplete
            ),
            "baseline_incomplete_count": len(
                strict_baseline_incomplete
            ),
            "legacy_complete_c3_complete_harm_question_ids": (
                strict_complete_to_complete_harm
            ),
            "legacy_complete_c3_complete_harm_count": len(
                strict_complete_to_complete_harm
            ),
        },
        "diagnostic_predictor_note": (
            "M2-C3.6 preservation features are post-hoc development "
            "diagnostics on the already-inspected 22-case semantic-harm "
            "subset. They must not be reported as unbiased final test "
            "performance. Any mechanism selected from this audit must be "
            "frozen and re-evaluated on held-out data."
        ),
        "support_retention_definition": (
            "|Legacy slot support ∩ C3 slot support| / "
            "|Legacy slot support|, evaluated on hard slots that are "
            "complete in both Legacy and C3."
        ),
        "weighted_retention_definition": {
            "requirement_gain": (
                "Legacy support retention weighted by existing "
                "selector_requirement_gain; no new learned weights."
            ),
            "topical_score": (
                "Legacy support retention weighted by existing "
                "selector_topical_score; no new learned weights."
            ),
        },
    }

    cases_path = (
        args.output_dir
        / "paired_slot_deficit_cases.csv"
    )

    slot_path = (
        args.output_dir
        / "slot_support_preservation_by_slot.csv"
    )

    stats_path = (
        args.output_dir
        / "support_preservation_feature_stats.csv"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    write_csv(
        cases_path,
        cases,
    )

    write_csv(
        slot_path,
        slot_rows,
    )

    write_csv(
        stats_path,
        stats,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "Diagnostic predictor results:"
    )

    for row in stats:
        print(
            f"- target={row['target']} "
            f"predictor={row['predictor']} "
            f"P={row['precision_for_harm']:.3f} "
            f"R={row['recall_for_harm']:.3f} "
            f"FPR={row['false_positive_rate']:.3f} "
            f"OR={row['odds_ratio_corrected']:.3f} "
            f"p={row['fisher_two_sided_p']:.4f}"
        )

    print()
    print(
        "Saved:",
        cases_path,
    )
    print(
        "Saved:",
        slot_path,
    )
    print(
        "Saved:",
        stats_path,
    )
    print(
        "Saved:",
        summary_path,
    )


if __name__ == "__main__":
    main()