from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    return []


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def classify(gold: set[str], raw: set[str], ranked: set[str], selected: set[str]) -> str:
    if not gold:
        return "no_gold"
    if not (gold & raw):
        return "retrieval_failure"
    if (gold & raw) and not (gold & ranked):
        return "ranking_failure"
    if (gold & ranked) and not (gold & selected):
        return "selection_failure"
    if gold.issubset(selected):
        return "all_gold_selected"
    return "partial_gold_selected"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictions = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    totals = Counter()
    loss_counts = Counter()

    with predictions.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            r = json.loads(line)

            qid = str(r.get("question_id", f"line_{line_no}"))
            query = str(r.get("query", ""))
            qtype = str(r.get("question_type", ""))

            gold_list = as_list(r.get("supporting_memory_ids"))
            raw_list = as_list(r.get("raw_retrieved_ids"))
            ranked_list = as_list(r.get("ranked_candidate_ids"))
            selected_list = as_list(r.get("selected_ids"))

            gold, raw = set(gold_list), set(raw_list)
            ranked, selected = set(ranked_list), set(selected_list)

            gold_raw = gold & raw
            gold_ranked = gold & ranked
            gold_selected = gold & selected

            stage = classify(gold, raw, ranked, selected)
            loss_counts[stage] += 1

            totals["questions"] += 1
            totals["gold"] += len(gold)
            totals["gold_raw"] += len(gold_raw)
            totals["gold_ranked"] += len(gold_ranked)
            totals["gold_selected"] += len(gold_selected)
            totals["raw_candidates"] += len(raw_list)
            totals["ranked_candidates"] += len(ranked_list)
            totals["selected_candidates"] += len(selected_list)

            stage_rows.append({
                "question_id": qid,
                "question_type": qtype,
                "query": query,
                "loss_stage": stage,
                "num_gold": len(gold),
                "gold_in_raw": len(gold_raw),
                "gold_in_ranked": len(gold_ranked),
                "gold_in_selected": len(gold_selected),
                "raw_recall": round(safe_div(len(gold_raw), len(gold)), 6),
                "ranked_recall": round(safe_div(len(gold_ranked), len(gold)), 6),
                "selected_recall": round(safe_div(len(gold_selected), len(gold)), 6),
                "ranking_retention": round(safe_div(len(gold_ranked), len(gold_raw)), 6),
                "selection_retention": round(safe_div(len(gold_selected), len(gold_ranked)), 6),
                "missing_from_raw": ";".join(sorted(gold - raw)),
                "lost_raw_to_ranked": ";".join(sorted(gold_raw - ranked)),
                "lost_ranked_to_selected": ";".join(sorted(gold_ranked - selected)),
                "gold_ids": ";".join(sorted(gold)),
                "raw_ids": ";".join(raw_list),
                "ranked_ids": ";".join(ranked_list),
                "selected_ids": ";".join(selected_list),
            })

            if bool(r.get("route_metric_applicable", True)):
                expected = set(as_list(r.get("expected_memory_types")))
                predicted = set(as_list(r.get("selected_memory_types")))
                exact = expected == predicted
                totals["route_scored"] += 1
                totals["route_exact"] += int(exact)
                if not exact:
                    route_rows.append({
                        "question_id": qid,
                        "question_type": qtype,
                        "query": query,
                        "query_mode": r.get("query_mode", ""),
                        "expected_memory_types": ";".join(sorted(expected)),
                        "predicted_memory_types": ";".join(sorted(predicted)),
                        "missing_memory_types": ";".join(sorted(expected - predicted)),
                        "extra_memory_types": ";".join(sorted(predicted - expected)),
                        "route_scores": json.dumps(r.get("route_scores", {}), ensure_ascii=False, sort_keys=True),
                    })

    summary = {
        "num_questions": totals["questions"],
        "total_gold_evidence": totals["gold"],
        "gold_in_raw": totals["gold_raw"],
        "gold_in_ranked": totals["gold_ranked"],
        "gold_in_selected": totals["gold_selected"],
        "raw_gold_recall": safe_div(totals["gold_raw"], totals["gold"]),
        "ranked_gold_recall": safe_div(totals["gold_ranked"], totals["gold"]),
        "selected_gold_recall": safe_div(totals["gold_selected"], totals["gold"]),
        "ranking_retention": safe_div(totals["gold_ranked"], totals["gold_raw"]),
        "selection_retention": safe_div(totals["gold_selected"], totals["gold_ranked"]),
        "average_raw_candidates": safe_div(totals["raw_candidates"], totals["questions"]),
        "average_ranked_candidates": safe_div(totals["ranked_candidates"], totals["questions"]),
        "average_selected_candidates": safe_div(totals["selected_candidates"], totals["questions"]),
        "route_scored": totals["route_scored"],
        "route_exact_count": totals["route_exact"],
        "route_exact_rate": safe_div(totals["route_exact"], totals["route_scored"]),
        "route_error_count": len(route_rows),
        "loss_stage_counts": dict(sorted(loss_counts.items())),
    }

    stage_path = output_dir / "stage_loss_report.csv"
    route_path = output_dir / "route_error_report.csv"
    summary_path = output_dir / "trace_audit_summary.json"

    write_csv(stage_path, stage_rows)
    write_csv(route_path, route_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("RC4 TRACE AUDIT")
    print("=" * 72)
    print(f"Gold in raw:       {totals['gold_raw']}/{totals['gold']} ({summary['raw_gold_recall']:.3f})")
    print(f"Gold in ranked:    {totals['gold_ranked']}/{totals['gold']} ({summary['ranked_gold_recall']:.3f})")
    print(f"Gold in selected:  {totals['gold_selected']}/{totals['gold']} ({summary['selected_gold_recall']:.3f})")
    print(f"Ranking retention: {summary['ranking_retention']:.3f}")
    print(f"Selection retention: {summary['selection_retention']:.3f}")
    print(f"Route exact:       {totals['route_exact']}/{totals['route_scored']} ({summary['route_exact_rate']:.3f})")
    print(f"Route errors:      {len(route_rows)}")
    print(f"Loss-stage counts: {summary['loss_stage_counts']}")
    print(f"Saved: {stage_path}")
    print(f"Saved: {route_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
