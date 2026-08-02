from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_memories(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
        return payload["memories"]
    raise TypeError(f"Unsupported memories JSON structure: {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return split_ids(decoded)
        except json.JSONDecodeError:
            pass
    return list(dict.fromkeys(item.strip() for item in text.split(";") if item.strip()))


def get_field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def build_source_map(memories: list[dict[str, Any]]) -> dict[str, list[str]]:
    source_map: dict[str, list[str]] = {}
    for record in memories:
        memory_id = str(get_field(record, "memory_id", "")).strip()
        if not memory_id:
            raise ValueError("Memory record has no memory_id")
        memory_type = str(get_field(record, "memory_type", "")).strip()
        source_ids = split_ids(get_field(record, "source_ids", []))
        canonical = source_ids if memory_type == "semantic" and source_ids else [memory_id]
        source_map[memory_id] = list(dict.fromkeys(canonical))
    return source_map


def project_ids(ids: list[str], source_map: dict[str, list[str]]) -> list[str]:
    projected: list[str] = []
    for item in ids:
        projected.extend(source_map.get(item, [item]))
    return list(dict.fromkeys(projected))


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score(
    questions: list[dict[str, str]],
    retrieval_rows: list[dict[str, Any]],
    source_map: dict[str, list[str]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    q_by_id = {row["question_id"]: row for row in questions}
    r_by_id = {str(row["question_id"]): row for row in retrieval_rows}

    if len(q_by_id) != len(questions):
        raise AssertionError("Duplicate question IDs in questions CSV")
    if len(r_by_id) != len(retrieval_rows):
        raise AssertionError("Duplicate question IDs in retrieval JSONL")
    if set(q_by_id) != set(r_by_id):
        raise AssertionError("Question IDs do not match between questions and retrieval rows")

    corrected: list[dict[str, Any]] = []
    metric_changed = 0
    gold_projection_changed = 0
    retrieval_projection_changed = 0

    for question in questions:
        qid = question["question_id"]
        old = r_by_id[qid]

        gold_memory_ids = split_ids(question.get("supporting_memory_ids", ""))
        retrieved_memory_ids = split_ids(old.get("retrieved_memory_ids", []))

        gold_source_ids = project_ids(gold_memory_ids, source_map)
        retrieved_source_ids = project_ids(retrieved_memory_ids, source_map)

        if gold_source_ids != split_ids(old.get("gold_ids", [])):
            gold_projection_changed += 1
        if retrieved_source_ids != split_ids(old.get("projected_source_ids", [])):
            retrieval_projection_changed += 1

        gold_set = set(gold_source_ids)
        retrieved_set = set(retrieved_source_ids)
        overlap = gold_set & retrieved_set

        precision = len(overlap) / len(retrieved_set) if retrieved_set else 0.0
        recall = len(overlap) / len(gold_set) if gold_set else 0.0
        f1 = harmonic_f1(precision, recall)

        old_metrics = (
            float(old.get("precision", 0.0)),
            float(old.get("recall", 0.0)),
            float(old.get("f1", 0.0)),
            bool(old.get("hit_at_k", False)),
        )
        new_metrics = (precision, recall, f1, bool(overlap))
        if old_metrics != new_metrics:
            metric_changed += 1

        row = dict(old)
        row.update(
            {
                "gold_memory_ids": gold_memory_ids,
                "gold_source_ids": gold_source_ids,
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_source_ids": retrieved_source_ids,
                "overlap_source_ids": sorted(overlap),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hit_at_k": bool(overlap),
                "retrieved_memory_count": len(retrieved_memory_ids),
                "projected_source_count": len(retrieved_source_ids),
                "scoring_mode": "source_equivalent_v02",
            }
        )
        corrected.append(row)

    n = len(corrected)
    summary = {
        "top_k": top_k,
        "questions": n,
        "scoring_mode": "source_equivalent_v02",
        "mean_precision": sum(row["precision"] for row in corrected) / n,
        "mean_recall": sum(row["recall"] for row in corrected) / n,
        "mean_f1": sum(row["f1"] for row in corrected) / n,
        "hit_at_k": sum(int(row["hit_at_k"]) for row in corrected) / n,
        "mean_retrieved_memory_count": sum(row["retrieved_memory_count"] for row in corrected) / n,
        "mean_projected_source_count": sum(row["projected_source_count"] for row in corrected) / n,
        "mean_latency_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in corrected) / n,
        "empty_retrievals": sum(int(row["retrieved_memory_count"] == 0) for row in corrected),
    }
    audit = {
        "top_k": top_k,
        "questions": n,
        "metric_changed_rows": metric_changed,
        "gold_projection_changed_rows": gold_projection_changed,
        "retrieval_projection_changed_rows": retrieval_projection_changed,
    }
    return corrected, summary, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score Mem0 Pilot-200 with corrected LoCoMo v0.2 source-equivalent evidence."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = read_csv(args.questions)
    memories = load_memories(args.memories)
    source_map = build_source_map(memories)

    if len(questions) != 200:
        raise AssertionError(f"Expected 200 questions, found {len(questions)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, Any] = {
        "pilot_name": "rc88_locomo_pilot200_mem0_v01",
        "pilot_questions": len(questions),
        "scoring_mode": "source_equivalent_v02",
        "configurations": {},
        "audits": {},
    }

    for top_k in (5, 20):
        retrieval_rows = read_jsonl(args.input_dir / f"retrieval_top{top_k}.jsonl")
        corrected, summary, audit = score(
            questions=questions,
            retrieval_rows=retrieval_rows,
            source_map=source_map,
            top_k=top_k,
        )

        write_jsonl(
            args.output_dir / f"retrieval_top{top_k}_source_equiv.jsonl",
            corrected,
        )
        (args.output_dir / f"summary_top{top_k}_source_equiv.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"audit_top{top_k}_source_equiv.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        combined["configurations"][f"top{top_k}"] = summary
        combined["audits"][f"top{top_k}"] = audit

        print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, indent=2))

    (args.output_dir / "retrieval_summary_source_equiv.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("MEM0 PILOT-200 SOURCE-EQUIVALENT RESCORING: PASSED")


if __name__ == "__main__":
    main()
