from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation_utils import score_prediction


METRICS = {
    "route_exact_match": "mean",
    "route_precision": "mean",
    "route_recall": "mean",
    "route_f1": "mean",
    "evidence_recall": "mean",
    "evidence_precision": "mean",
    "evidence_density": "mean",
    "include_score": "mean",
    "not_include_violations": "mean",
    "abstention_correctness": "mean",
    "confidence_score": "mean",
    "query_coverage": "mean",
    "latency_seconds": "mean",
    "num_retrieved": "mean",
    "num_used": "mean",
    "num_outdated": "mean",
    "conflict_action": "mean",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(predictions_path)
    scores: list[dict] = []

    for record in predictions:
        rescored_record = dict(record)
        rescored_record["expected_abstention"] = record.get(
            "should_abstain",
            not bool(record.get("supporting_memory_ids", [])),
        )
        scores.append(score_prediction(rescored_record))

    score_df = pd.DataFrame(scores)
    score_df.to_csv(output_dir / "scores.csv", index=False)

    available_metrics = {
        metric: aggregation
        for metric, aggregation in METRICS.items()
        if metric in score_df.columns
    }

    summary = (
        score_df.groupby("method")
        .agg(available_metrics)
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    by_type = (
        score_df.groupby(["question_type", "method"])
        .agg(available_metrics)
        .reset_index()
    )
    by_type.to_csv(
        output_dir / "summary_by_question_type.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    print()
    print(
        "Rescored with v2.1 abstention semantics: explicit should_abstain "
        "when available, otherwise empty supporting_memory_ids means abstain."
    )
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
