from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


CITATION_PATTERN = re.compile(
    r"\(?\[[esp]_[A-Za-z0-9_.:-]+\]\)?",
    flags=re.IGNORECASE,
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def normalize_answer(text: object) -> str:
    value = str(text or "").lower()
    value = CITATION_PATTERN.sub(" ", value)

    value = "".join(
        character
        if character not in string.punctuation
        else " "
        for character in value
    )

    value = re.sub(
        r"\b(a|an|the)\b",
        " ",
        value,
    )

    return " ".join(value.split())


def token_f1(
    prediction: object,
    gold: object,
) -> float:
    prediction_tokens = normalize_answer(
        prediction
    ).split()
    gold_tokens = normalize_answer(
        gold
    ).split()

    if not prediction_tokens and not gold_tokens:
        return 1.0

    if not prediction_tokens or not gold_tokens:
        return 0.0

    prediction_counts = Counter(
        prediction_tokens
    )
    gold_counts = Counter(gold_tokens)

    overlap = sum(
        (
            prediction_counts
            & gold_counts
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(
        prediction_tokens
    )
    recall = overlap / len(gold_tokens)

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    scored_rows: list[dict[str, object]] = []

    for row in rows:
        prediction = row.get("answer", "")
        gold = row.get("gold_answer", "")

        normalized_prediction = (
            normalize_answer(prediction)
        )
        normalized_gold = normalize_answer(
            gold
        )

        scored_rows.append(
            {
                "method": row.get("method"),
                "question_id": row.get(
                    "question_id"
                ),
                "question_type": row.get(
                    "question_type"
                ),
                "should_abstain": parse_bool(
                    row.get(
                        "should_abstain",
                        False,
                    )
                ),
                "answer_em": float(
                    normalized_prediction
                    == normalized_gold
                ),
                "answer_token_f1": token_f1(
                    prediction,
                    gold,
                ),
                "prediction": prediction,
                "gold_answer": gold,
            }
        )

    by_method: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    by_method_type: dict[
        tuple[str, str],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in scored_rows:
        method = str(row["method"])
        question_type = str(
            row["question_type"]
        )

        by_method[method].append(row)
        by_method_type[
            (method, question_type)
        ].append(row)

    summary_rows = []

    for method, values in sorted(
        by_method.items()
    ):
        answerable = [
            row
            for row in values
            if not bool(row["should_abstain"])
        ]

        summary_rows.append(
            {
                "method": method,
                "num_questions": len(values),
                "num_answerable": len(answerable),
                "answer_em_all": mean(
                    float(row["answer_em"])
                    for row in values
                ),
                "answer_f1_all": mean(
                    float(
                        row["answer_token_f1"]
                    )
                    for row in values
                ),
                "answer_em_answerable": mean(
                    float(row["answer_em"])
                    for row in answerable
                )
                if answerable
                else 0.0,
                "answer_f1_answerable": mean(
                    float(
                        row["answer_token_f1"]
                    )
                    for row in answerable
                )
                if answerable
                else 0.0,
            }
        )

    type_rows = []

    for (
        method,
        question_type,
    ), values in sorted(
        by_method_type.items()
    ):
        type_rows.append(
            {
                "method": method,
                "question_type": question_type,
                "num_questions": len(values),
                "answer_em": mean(
                    float(row["answer_em"])
                    for row in values
                ),
                "answer_token_f1": mean(
                    float(
                        row["answer_token_f1"]
                    )
                    for row in values
                ),
            }
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = [
        (
            "answer_quality_summary.csv",
            summary_rows,
        ),
        (
            "answer_quality_by_type.csv",
            type_rows,
        ),
        (
            "answer_quality_per_question.csv",
            scored_rows,
        ),
    ]

    for filename, output_rows in outputs:
        path = args.output_dir / filename

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    output_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(output_rows)

        print(path)


if __name__ == "__main__":
    main()
