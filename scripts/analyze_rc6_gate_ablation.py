from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def admitted(
    row: dict[str, str],
    *,
    utility_threshold: float,
    lexical_threshold: float,
    entity_threshold: float,
    temporal_min: float,
    validity_min: float,
    route_min: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if row.get("memory_type") == "procedural":
        reasons.append("procedural_safety_gate")
    if as_float(row.get("lexical_score")) >= lexical_threshold:
        reasons.append("lexical_gate")
    if as_float(row.get("graph_entity_score")) >= entity_threshold:
        reasons.append("entity_gate")
    if as_float(row.get("candidate_utility_v0")) >= utility_threshold:
        reasons.append("utility_gate")
    if (
        row.get("memory_type") in {"episodic", "semantic"}
        and as_float(row.get("temporal_task_score")) >= temporal_min
        and as_float(row.get("validity_score")) >= validity_min
        and as_float(row.get("route_compatibility_score")) >= route_min
    ):
        reasons.append("temporal_route_rescue")

    return bool(reasons), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.375,0.39,0.40,0.425,0.45,0.475,0.50",
    )
    parser.add_argument("--lexical-threshold", type=float, default=0.02)
    parser.add_argument("--entity-threshold", type=float, default=0.20)
    parser.add_argument("--temporal-min", type=float, default=0.95)
    parser.add_argument("--validity-min", type=float, default=0.85)
    parser.add_argument("--route-min", type=float, default=0.50)
    args = parser.parse_args()

    thresholds = [
        float(value)
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    with args.candidate_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    total_candidates = len(rows)
    total_gold = sum(as_bool(row.get("is_gold")) for row in rows)
    output_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        kept: list[dict[str, str]] = []
        gold_kept = 0
        rescued_gold = 0
        reason_counts: dict[str, int] = {}

        for row in rows:
            keep, reasons = admitted(
                row,
                utility_threshold=threshold,
                lexical_threshold=args.lexical_threshold,
                entity_threshold=args.entity_threshold,
                temporal_min=args.temporal_min,
                validity_min=args.validity_min,
                route_min=args.route_min,
            )
            if not keep:
                continue

            kept.append(row)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if as_bool(row.get("is_gold")):
                gold_kept += 1
                if not as_bool(row.get("passed_ranker_gate")):
                    rescued_gold += 1

        output_rows.append(
            {
                "utility_threshold": threshold,
                "candidate_count": len(kept),
                "candidate_retention": round(
                    len(kept) / max(total_candidates, 1),
                    6,
                ),
                "gold_kept": gold_kept,
                "gold_recall": round(gold_kept / max(total_gold, 1), 6),
                "gold_precision": round(gold_kept / max(len(kept), 1), 6),
                "previously_gate_lost_gold_rescued": rescued_gold,
                "utility_gate_count": reason_counts.get("utility_gate", 0),
                "temporal_rescue_count": reason_counts.get(
                    "temporal_route_rescue",
                    0,
                ),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 96)
    print("RC6 HYBRID UTILITY-AWARE GATE ABLATION")
    print("=" * 96)
    for row in output_rows:
        print(
            "threshold={utility_threshold:.3f}  candidates={candidate_count:>4}  "
            "retention={candidate_retention:.3f}  gold={gold_kept:>2}/{total_gold}  "
            "recall={gold_recall:.3f}  rescued={previously_gate_lost_gold_rescued}".format(
                total_gold=total_gold,
                **row,
            )
        )
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()
