from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

QUESTION_ID_COLUMNS = ("question_id", "query_id", "id")
GOLD_COLUMNS = (
    # Dataset A / run_experiment.py field used by the existing RC4 audit.
    "supporting_memory_ids",

    # Compatible alternative names.
    "gold_evidence_ids",
    "gold_ids",
    "gold_memory_ids",
    "expected_evidence_ids",
    "evidence_ids",
    "gold_evidence",
    "gold_source_ids",
)


def _normalise_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return _normalise_ids(parsed)

    separator = ";" if ";" in text else ","
    return [
        item.strip().strip("'\"")
        for item in text.split(separator)
        if item.strip().strip("'\"")
    ]


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _load_gold_dataset(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {}

    output: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            question_id = _first_present(row, QUESTION_ID_COLUMNS)
            if question_id is None:
                continue
            gold_value = _first_present(row, GOLD_COLUMNS)
            output[str(question_id)] = set(_normalise_ids(gold_value))
    return output


def _prediction_payload(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("result", "prediction", "output"):
        value = row.get(key)
        if isinstance(value, dict):
            merged = dict(row)
            merged.update(value)
            return merged
    return row


def _extract_question_id(row: dict[str, Any]) -> str:
    value = _first_present(row, QUESTION_ID_COLUMNS)
    return str(value) if value is not None else ""


def _extract_gold_ids(
    row: dict[str, Any],
    dataset_gold: dict[str, set[str]],
) -> set[str]:
    question_id = _extract_question_id(row)
    if question_id in dataset_gold:
        return dataset_gold[question_id]
    return set(_normalise_ids(_first_present(row, GOLD_COLUMNS)))


def analyse(
    *,
    predictions_path: Path,
    dataset_path: Path | None,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_gold = _load_gold_dataset(dataset_path)

    candidate_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    stage_counter: Counter[str] = Counter()

    total_questions = 0
    total_gold = 0
    gold_in_trace = 0
    gold_passed_gate = 0
    gold_kept_top_k = 0
    gold_survived_resolution = 0
    gold_selected = 0

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            raw_row = json.loads(line)
            row = _prediction_payload(raw_row)
            debug = row.get("debug", {})
            if not isinstance(debug, dict):
                debug = {}
            trace = debug.get("full_candidate_score_trace", [])
            if not isinstance(trace, list):
                trace = []

            question_id = _extract_question_id(row)
            query = str(row.get("query", ""))
            gold_ids = _extract_gold_ids(row, dataset_gold)

            total_questions += 1
            total_gold += len(gold_ids)

            trace_by_id = {
                str(item.get("memory_id")): item
                for item in trace
                if isinstance(item, dict) and item.get("memory_id") is not None
            }

            for item in trace:
                if not isinstance(item, dict):
                    continue
                memory_id = str(item.get("memory_id", ""))
                candidate_rows.append({
                    "question_id": question_id,
                    "query": query,
                    "is_gold": memory_id in gold_ids,
                    **item,
                })

            for gold_id in sorted(gold_ids):
                item = trace_by_id.get(gold_id)
                if item is None:
                    stage = "not_in_raw_trace"
                    gold_row = {
                        "question_id": question_id,
                        "query": query,
                        "gold_id": gold_id,
                        "memory_type": "",
                        "candidate_utility_v0": "",
                        "rank_global_all_scored": "",
                        "rank_within_type_all_scored": "",
                        "passed_ranker_gate": False,
                        "rank_within_type_after_gate": "",
                        "kept_after_type_top_k": False,
                        "survived_conflict_resolution": False,
                        "selected_final": False,
                        "drop_stage": stage,
                    }
                else:
                    gold_in_trace += 1
                    passed_gate = bool(item.get("passed_ranker_gate"))
                    kept_top_k = bool(item.get("kept_after_type_top_k"))
                    survived_resolution = bool(
                        item.get("survived_conflict_resolution")
                    )
                    selected_final = bool(item.get("selected_final"))

                    gold_passed_gate += int(passed_gate)
                    gold_kept_top_k += int(kept_top_k)
                    gold_survived_resolution += int(survived_resolution)
                    gold_selected += int(selected_final)
                    stage = item.get("drop_stage") or "selected"

                    gold_row = {
                        "question_id": question_id,
                        "query": query,
                        "gold_id": gold_id,
                        "memory_type": item.get("memory_type", ""),
                        "candidate_utility_v0": item.get(
                            "candidate_utility_v0", ""
                        ),
                        "rank_global_all_scored": item.get(
                            "rank_global_all_scored", ""
                        ),
                        "rank_within_type_all_scored": item.get(
                            "rank_within_type_all_scored", ""
                        ),
                        "passed_ranker_gate": passed_gate,
                        "rank_within_type_after_gate": item.get(
                            "rank_within_type_after_gate", ""
                        ),
                        "kept_after_type_top_k": kept_top_k,
                        "survived_conflict_resolution": survived_resolution,
                        "selected_final": selected_final,
                        "drop_stage": stage,
                    }

                stage_counter[str(stage)] += 1
                gold_rows.append(gold_row)

    candidate_columns = sorted({key for row in candidate_rows for key in row})
    with (output_dir / "all_candidate_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_columns)
        writer.writeheader()
        writer.writerows(candidate_rows)

    gold_columns = [
        "question_id",
        "query",
        "gold_id",
        "memory_type",
        "candidate_utility_v0",
        "rank_global_all_scored",
        "rank_within_type_all_scored",
        "passed_ranker_gate",
        "rank_within_type_after_gate",
        "kept_after_type_top_k",
        "survived_conflict_resolution",
        "selected_final",
        "drop_stage",
    ]
    with (output_dir / "gold_candidate_diagnostics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=gold_columns)
        writer.writeheader()
        writer.writerows(gold_rows)

    if total_gold == 0:
        available_prediction_keys: list[str] = []

        with predictions_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if line.strip():
                    first_row = _prediction_payload(
                        json.loads(line)
                    )
                    available_prediction_keys = sorted(
                        first_row.keys()
                    )
                    break

        raise ValueError(
            "No gold evidence IDs were found. "
            "Expected one of the supported fields: "
            f"{', '.join(GOLD_COLUMNS)}. "
            "Available prediction fields include: "
            f"{available_prediction_keys}"
        )

    denominator = total_gold
    summary = {
        "num_questions": total_questions,
        "total_gold_evidence": total_gold,
        "gold_in_raw_trace": gold_in_trace,
        "gold_passed_ranker_gate": gold_passed_gate,
        "gold_kept_after_type_top_k": gold_kept_top_k,
        "gold_survived_conflict_resolution": gold_survived_resolution,
        "gold_selected_final": gold_selected,
        "raw_trace_recall": gold_in_trace / denominator,
        "ranker_gate_recall": gold_passed_gate / denominator,
        "type_top_k_recall": gold_kept_top_k / denominator,
        "conflict_resolution_recall": (
            gold_survived_resolution / denominator
        ),
        "final_selection_recall": gold_selected / denominator,
        "gold_drop_stage_counts": dict(stage_counter),
        "num_candidate_rows": len(candidate_rows),
    }

    with (output_dir / "rc5_candidate_trace_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("RC5 FULL CANDIDATE TRACE AUDIT")
    print("=" * 72)
    print(
        f"Gold in raw trace:       {gold_in_trace}/{total_gold} "
        f"({summary['raw_trace_recall']:.3f})"
    )
    print(
        f"Gold passed ranker gate: {gold_passed_gate}/{total_gold} "
        f"({summary['ranker_gate_recall']:.3f})"
    )
    print(
        f"Gold kept by type top-k: {gold_kept_top_k}/{total_gold} "
        f"({summary['type_top_k_recall']:.3f})"
    )
    print(
        f"Gold after resolution:   {gold_survived_resolution}/{total_gold} "
        f"({summary['conflict_resolution_recall']:.3f})"
    )
    print(
        f"Gold selected final:     {gold_selected}/{total_gold} "
        f"({summary['final_selection_recall']:.3f})"
    )
    print("Drop stages:", dict(stage_counter))
    print("Saved:", output_dir / "all_candidate_scores.csv")
    print("Saved:", output_dir / "gold_candidate_diagnostics.csv")
    print("Saved:", output_dir / "rc5_candidate_trace_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    analyse(
        predictions_path=args.predictions,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
