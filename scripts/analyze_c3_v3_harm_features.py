from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_ids(value: Any) -> list[str]:
    return [str(x) for x in as_list(value) if str(x).strip()]


def memory_type_from_id(memory_id: str) -> str:
    if memory_id.startswith("p_"):
        return "procedural"
    if memory_id.startswith("e_"):
        return "episodic"
    if memory_id.startswith("s_"):
        return "semantic"
    return "unknown"


def candidate_trace_index(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace = (row.get("debug") or {}).get("full_candidate_score_trace") or []
    output: dict[str, dict[str, Any]] = {}

    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", ""))
            if memory_id:
                output[memory_id] = item

    selected_evidence = row.get("selected_evidence") or []
    if isinstance(selected_evidence, list):
        for item in selected_evidence:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", ""))
            if not memory_id:
                continue
            merged = dict(output.get(memory_id, {}))
            merged.update(item)
            output[memory_id] = merged

    return output


def get_roles(trace_item: dict[str, Any]) -> set[str]:
    raw = (
        trace_item.get("evidence_roles")
        or trace_item.get("roles")
        or trace_item.get("matched_roles")
        or []
    )
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw}
    return set()


def temporal_role(trace_item: dict[str, Any]) -> str:
    return str(
        trace_item.get("query_relative_temporal_role")
        or trace_item.get("temporal_role")
        or ""
    )


def temporal_compatible(trace_item: dict[str, Any]) -> bool | None:
    if "query_relative_temporal_compatible" in trace_item:
        return bool(trace_item["query_relative_temporal_compatible"])
    if "temporal_compatible" in trace_item:
        return bool(trace_item["temporal_compatible"])
    return None


def info_need_gain_for(
    shadow: dict[str, Any],
    memory_id: str,
) -> float:
    table = shadow.get("removed_information_need_gain") or {}
    if not isinstance(table, dict):
        return 0.0
    item = table.get(memory_id) or {}
    if not isinstance(item, dict):
        return 0.0
    return float(item.get("gain", 0.0) or 0.0)


def query_is_enumerative(query: str) -> bool:
    text = query.lower().strip()

    patterns = [
        r"\bwhich\s+(abilities|skills|details|factors|components|parts|features|types|aspects|items|things)\b",
        r"\bwhat\s+(abilities|skills|details|factors|components|parts|features|types|aspects|items|things)\b",
        r"\bwhat\s+technical\s+details\b",
        r"\bhow\s+did\b.*\bchange\b",
        r"\bhow\s+did\b.*\bchange\s+over\s+time\b",
        r"\bwhat\s+changed\b",
        r"\blist\b",
        r"\benumerate\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def query_requests_explainability(
    question_type: str,
    query: str,
) -> bool:
    text = query.lower()
    return (
        question_type == "explainability"
        or "explain" in text
        or "which memories" in text
        or "memory types" in text
        or "cite the memory" in text
        or "cite memory" in text
    )


def selected_role_count(
    selected_ids: list[str],
    trace: dict[str, dict[str, Any]],
    role: str,
) -> int:
    return sum(
        role in get_roles(trace.get(memory_id, {}))
        for memory_id in selected_ids
    )


def selected_temporal_role_count(
    selected_ids: list[str],
    trace: dict[str, dict[str, Any]],
    role: str,
) -> int:
    return sum(
        temporal_role(trace.get(memory_id, {})) == role
        for memory_id in selected_ids
    )


def safe_bool(value: Any) -> bool:
    return bool(value)


def outcome(delta: float) -> str:
    if delta < -EPS:
        return "worse"
    if delta > EPS:
        return "better"
    return "tied"


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test for a 2x2 table without scipy."""

    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def probability(x: int) -> float:
        y = row1 - x
        z = col1 - x
        w = row2 - z

        if min(x, y, z, w) < 0:
            return 0.0

        return (
            math.comb(col1, x)
            * math.comb(total - col1, row1 - x)
            / math.comb(total, row1)
        )

    low = max(0, row1 - (total - col1))
    high = min(row1, col1)

    observed = probability(a)
    p_value = 0.0

    for x in range(low, high + 1):
        p = probability(x)
        if p <= observed + 1e-15:
            p_value += p

    return min(1.0, p_value)


def binary_feature_stats(
    cases: list[dict[str, Any]],
    feature: str,
) -> dict[str, Any]:
    harmful = [case for case in cases if case["outcome"] == "worse"]
    harmless = [case for case in cases if case["outcome"] != "worse"]

    a = sum(bool(case[feature]) for case in harmful)
    b = sum(bool(case[feature]) for case in harmless)
    c = len(harmful) - a
    d = len(harmless) - b

    precision = a / (a + b) if (a + b) else 0.0
    recall = a / len(harmful) if harmful else 0.0
    false_positive_rate = b / len(harmless) if harmless else 0.0
    specificity = d / len(harmless) if harmless else 0.0

    # Haldane-Anscombe correction makes the odds ratio finite when a cell is 0.
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

    return {
        "feature": feature,
        "harmful_flagged": a,
        "harmful_not_flagged": c,
        "harmless_flagged": b,
        "harmless_not_flagged": d,
        "precision_for_harm": round(precision, 6),
        "recall_for_harm": round(recall, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "specificity": round(specificity, 6),
        "odds_ratio_corrected": round(odds_ratio, 6),
        "fisher_two_sided_p": round(fisher_two_sided(a, b, c, d), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generation-time harmful vs harmless C3-v3 pruning "
            "among questions where annotated gold evidence was removed."
        )
    )
    parser.add_argument(
        "--shadow-predictions",
        type=Path,
        required=True,
        help="Dataset A Full60 C3-v3 shadow predictions.jsonl",
    )
    parser.add_argument(
        "--generation-predictions",
        type=Path,
        required=True,
        help="Qwen paired counterfactual predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    shadow_rows = {
        str(row.get("question_id", "")): row
        for row in read_jsonl(args.shadow_predictions)
        if str(row.get("method", "")).lower() == "c3"
    }

    generation_rows = read_jsonl(args.generation_predictions)
    generation_by_q: dict[str, dict[str, dict[str, Any]]] = {}

    for row in generation_rows:
        qid = str(row.get("question_id", ""))
        variant = str(row.get("variant", ""))
        if variant not in {"legacy", "c3_v3"}:
            continue
        generation_by_q.setdefault(qid, {})[variant] = row

    cases: list[dict[str, Any]] = []

    for qid, variants in sorted(generation_by_q.items()):
        if "legacy" not in variants or "c3_v3" not in variants:
            continue

        legacy_gen = variants["legacy"]
        c3_gen = variants["c3_v3"]

        removed_gold_ids = as_ids(c3_gen.get("removed_gold_ids"))
        if not removed_gold_ids:
            continue

        shadow_row = shadow_rows.get(qid)
        if shadow_row is None:
            raise AssertionError(f"{qid}: missing shadow prediction")

        shadow = (
            (shadow_row.get("debug") or {})
            .get("c3_v3_arbitration_shadow")
        )
        if not isinstance(shadow, dict):
            raise AssertionError(f"{qid}: missing c3_v3_arbitration_shadow")

        trace = candidate_trace_index(shadow_row)

        legacy_ids = as_ids(shadow.get("legacy_selected_ids"))
        c3_ids = as_ids(shadow.get("c3_v3_selected_ids"))
        removed_ids = as_ids(shadow.get("removed_by_c3_v3"))

        delta = (
            float(c3_gen.get("answer_f1", 0.0) or 0.0)
            - float(legacy_gen.get("answer_f1", 0.0) or 0.0)
        )

        question_type = str(shadow_row.get("question_type", ""))
        query_mode = str(shadow_row.get("query_mode", ""))
        query = str(shadow_row.get("query", ""))

        c3_types = [memory_type_from_id(memory_id) for memory_id in c3_ids]
        legacy_types = [memory_type_from_id(memory_id) for memory_id in legacy_ids]
        removed_gold_types = [
            memory_type_from_id(memory_id)
            for memory_id in removed_gold_ids
        ]

        removed_gold_roles: set[str] = set()
        removed_gold_temporal_roles: set[str] = set()
        removed_gold_any_incompatible = False
        removed_gold_positive_need_gain = False

        for memory_id in removed_gold_ids:
            item = trace.get(memory_id, {})
            removed_gold_roles |= get_roles(item)

            temp_role = temporal_role(item)
            if temp_role:
                removed_gold_temporal_roles.add(temp_role)

            if temporal_compatible(item) is False:
                removed_gold_any_incompatible = True

            if info_need_gain_for(shadow, memory_id) > 0.0:
                removed_gold_positive_need_gain = True

        c3_answer_target_count = selected_role_count(
            c3_ids, trace, "answer_target"
        )
        legacy_answer_target_count = selected_role_count(
            legacy_ids, trace, "answer_target"
        )

        c3_supporting_count = selected_role_count(
            c3_ids, trace, "supporting_evidence"
        )
        legacy_supporting_count = selected_role_count(
            legacy_ids, trace, "supporting_evidence"
        )

        # Safeguard 1: removed answer-bearing gold, while final set has no
        # candidate explicitly tagged answer_target.
        direct_answer_anchor_risk = (
            "answer_target" in removed_gold_roles
            and c3_answer_target_count == 0
        )

        c3_nonprocedural_count = sum(
            memory_type != "procedural"
            for memory_type in c3_types
        )
        c3_procedural_count = sum(
            memory_type == "procedural"
            for memory_type in c3_types
        )

        # Safeguard 2: policy/procedure survives while factual grounding was
        # removed. This signal is fully available before generation.
        procedure_fact_complementarity_risk = (
            c3_procedural_count > 0
            and c3_nonprocedural_count == 0
            and any(
                memory_type != "procedural"
                for memory_type in removed_gold_types
            )
        )

        # Safeguard 3: temporal structural evidence has been lost.
        if query_mode == "timeline":
            c3_transition_count = selected_temporal_role_count(
                c3_ids, trace, "transition_event"
            )
            temporal_structure_risk = (
                "transition_event" in removed_gold_temporal_roles
                and c3_transition_count == 0
            )
        elif query_mode == "current_state":
            c3_current_count = selected_temporal_role_count(
                c3_ids, trace, "current_state"
            )
            temporal_structure_risk = (
                c3_current_count == 0
                and (
                    "current_state" in removed_gold_temporal_roles
                    or "answer_target" in removed_gold_roles
                )
            )
        else:
            temporal_structure_risk = False

        # Safeguard 4: query asks for multiple facets but arbitration collapsed
        # the answer-bearing set substantially.
        enumerative_query = query_is_enumerative(query)
        multi_facet_cardinality_risk = (
            enumerative_query
            and legacy_answer_target_count >= 2
            and c3_answer_target_count < legacy_answer_target_count
        )

        # Safeguard 5: explainability requires both an answer and support /
        # provenance, but C3 has collapsed that structure.
        explainability_query = query_requests_explainability(
            question_type,
            query,
        )
        c3_type_diversity = len(set(c3_types))
        legacy_type_diversity = len(set(legacy_types))

        explainability_provenance_risk = (
            explainability_query
            and (
                (
                    legacy_supporting_count > 0
                    and c3_supporting_count == 0
                )
                or (
                    legacy_type_diversity >= 2
                    and c3_type_diversity < legacy_type_diversity
                    and len(c3_ids) <= 1
                )
            )
        )

        safeguards = {
            "sg_direct_answer_anchor": direct_answer_anchor_risk,
            "sg_procedure_fact_complementarity": (
                procedure_fact_complementarity_risk
            ),
            "sg_temporal_structure": temporal_structure_risk,
            "sg_multi_facet_cardinality": multi_facet_cardinality_risk,
            "sg_explainability_provenance": explainability_provenance_risk,
        }

        safeguard_count = sum(safeguards.values())
        safeguard_union = safeguard_count > 0

        cases.append(
            {
                "question_id": qid,
                "question_type": question_type,
                "query_mode": query_mode,
                "query": query,
                "outcome": outcome(delta),
                "answer_f1_delta": round(delta, 6),
                "legacy_answer_f1": round(
                    float(legacy_gen.get("answer_f1", 0.0) or 0.0),
                    6,
                ),
                "c3_v3_answer_f1": round(
                    float(c3_gen.get("answer_f1", 0.0) or 0.0),
                    6,
                ),
                "removed_gold_count": len(removed_gold_ids),
                "removed_gold_ids": ";".join(removed_gold_ids),
                "removed_all_count": len(removed_ids),
                "legacy_selected_count": len(legacy_ids),
                "c3_v3_selected_count": len(c3_ids),
                "selection_count_delta": len(c3_ids) - len(legacy_ids),
                "legacy_answer_target_count": legacy_answer_target_count,
                "c3_answer_target_count": c3_answer_target_count,
                "legacy_supporting_count": legacy_supporting_count,
                "c3_supporting_count": c3_supporting_count,
                "legacy_memory_type_diversity": legacy_type_diversity,
                "c3_memory_type_diversity": c3_type_diversity,
                "c3_procedural_count": c3_procedural_count,
                "c3_nonprocedural_count": c3_nonprocedural_count,
                "hard_complete": safe_bool(shadow.get("hard_complete")),
                "soft_complete": safe_bool(shadow.get("soft_complete")),
                "legacy_information_need_coverage": float(
                    shadow.get("legacy_information_need_coverage", 0.0) or 0.0
                ),
                "c3_information_need_coverage": float(
                    shadow.get("c3_v3_information_need_coverage", 0.0) or 0.0
                ),
                "information_need_coverage_delta": float(
                    shadow.get("information_need_coverage_delta", 0.0) or 0.0
                ),
                "removed_gold_positive_need_gain": (
                    removed_gold_positive_need_gain
                ),
                "removed_gold_any_incompatible": (
                    removed_gold_any_incompatible
                ),
                "removed_gold_has_answer_target": (
                    "answer_target" in removed_gold_roles
                ),
                "removed_gold_has_supporting": (
                    "supporting_evidence" in removed_gold_roles
                ),
                "removed_gold_has_transition": (
                    "transition_event" in removed_gold_temporal_roles
                ),
                "enumerative_query": enumerative_query,
                "explainability_query": explainability_query,
                **safeguards,
                "sg_union": safeguard_union,
                "sg_count": safeguard_count,
                "legacy_answer": str(legacy_gen.get("answer", "")),
                "c3_v3_answer": str(c3_gen.get("answer", "")),
                "gold_answer": str(c3_gen.get("gold_answer", "")),
            }
        )

    if len(cases) != 22:
        raise AssertionError(
            f"Expected 22 removed-gold questions from the current audit, got {len(cases)}."
        )

    outcome_counts = Counter(case["outcome"] for case in cases)

    binary_features = [
        "hard_complete",
        "soft_complete",
        "removed_gold_positive_need_gain",
        "removed_gold_any_incompatible",
        "removed_gold_has_answer_target",
        "removed_gold_has_supporting",
        "removed_gold_has_transition",
        "enumerative_query",
        "explainability_query",
        "sg_direct_answer_anchor",
        "sg_procedure_fact_complementarity",
        "sg_temporal_structure",
        "sg_multi_facet_cardinality",
        "sg_explainability_provenance",
        "sg_union",
    ]

    feature_stats = [
        binary_feature_stats(cases, feature)
        for feature in binary_features
    ]

    feature_stats.sort(
        key=lambda row: (
            -float(row["recall_for_harm"]),
            -float(row["precision_for_harm"]),
            float(row["false_positive_rate"]),
        )
    )

    harmful_cases = [case for case in cases if case["outcome"] == "worse"]
    harmless_cases = [case for case in cases if case["outcome"] != "worse"]

    union_stats = binary_feature_stats(cases, "sg_union")

    summary = {
        "questions_with_removed_gold": len(cases),
        "outcomes": dict(outcome_counts),
        "harmful_definition": "c3_v3_answer_f1 < legacy_answer_f1",
        "harmful_questions": len(harmful_cases),
        "harmless_questions": len(harmless_cases),
        "safeguard_union": union_stats,
        "safeguard_flag_counts": {
            feature: sum(bool(case[feature]) for case in cases)
            for feature in [
                "sg_direct_answer_anchor",
                "sg_procedure_fact_complementarity",
                "sg_temporal_structure",
                "sg_multi_facet_cardinality",
                "sg_explainability_provenance",
            ]
        },
        "notes": [
            (
                "This audit uses observed token-level Answer F1 deltas as the "
                "objective harm label. It does not assume that every negative "
                "delta is a semantic failure."
            ),
            (
                "Safeguard features are deterministic and intended to be "
                "computable before generation; no classifier is trained."
            ),
            (
                "A safeguard should not be activated from this audit alone. "
                "Inspect false positives and semantic-noise cases before "
                "modifying EvidenceArbitratorV3."
            ),
        ],
    }

    write_csv(
        args.output_dir / "harm_feature_cases.csv",
        cases,
    )
    write_csv(
        args.output_dir / "binary_feature_stats.csv",
        feature_stats,
    )

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Top binary features:")
    for row in feature_stats[:10]:
        print(
            f"- {row['feature']}: "
            f"recall={row['recall_for_harm']:.3f}, "
            f"precision={row['precision_for_harm']:.3f}, "
            f"FPR={row['false_positive_rate']:.3f}, "
            f"OR={row['odds_ratio_corrected']:.3f}, "
            f"p={row['fisher_two_sided_p']:.4f}"
        )


if __name__ == "__main__":
    main()
