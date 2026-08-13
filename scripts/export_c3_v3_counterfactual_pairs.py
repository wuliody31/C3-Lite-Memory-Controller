from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported JSON root in {path}")

    for key in ("memories", "procedures", "records", "items"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    for value in data.values():
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value

    raise ValueError(f"No record list found in {path}")


def record_id(record: dict[str, Any]) -> str:
    for key in (
        "memory_id",
        "procedure_id",
        "rule_id",
        "id",
    ):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def trace_index(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace = (row.get("debug") or {}).get("full_candidate_score_trace") or []
    output: dict[str, dict[str, Any]] = {}
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            memory_id = str(item.get("memory_id", ""))
            if memory_id:
                output[memory_id] = item
    return output


def public_evidence(
    memory_id: str,
    catalog: dict[str, dict[str, Any]],
    trace: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = dict(catalog.get(memory_id, {}))
    score_trace = trace.get(memory_id, {})

    memory_type = (
        record.get("memory_type")
        or record.get("type")
        or (
            "procedural"
            if memory_id.startswith("p_")
            else "episodic"
            if memory_id.startswith("e_")
            else "semantic"
            if memory_id.startswith("s_")
            else "unknown"
        )
    )

    text = (
        record.get("text")
        or record.get("instruction")
        or record.get("content")
        or ""
    )

    return {
        "memory_id": memory_id,
        "memory_type": str(memory_type),
        "text": str(text),
        "timestamp": record.get("timestamp"),
        "subject": record.get("subject"),
        "predicate": record.get("predicate"),
        "object_value": record.get("object_value"),
        "temporal_role": score_trace.get("query_relative_temporal_role"),
        "temporal_compatible": score_trace.get("query_relative_temporal_compatible"),
    }


def make_prompt(*, query: str, evidence: list[dict[str, Any]]) -> str:
    lines = [
        "SYSTEM",
        "",
        "You are answering a question using only the supplied memory evidence.",
        "Do not use outside facts.",
        "Do not invent unsupported details.",
        "Treat historical and current evidence according to their stated role.",
        "If the evidence is insufficient, say that the stored evidence is insufficient.",
        "",
        "USER QUERY",
        "",
        query,
        "",
        "MEMORY EVIDENCE",
        "",
    ]

    if not evidence:
        lines.append("- None")
    else:
        for item in evidence:
            metadata = []
            if item.get("timestamp"):
                metadata.append(f"time={item['timestamp']}")
            if item.get("temporal_role"):
                metadata.append(f"role={item['temporal_role']}")

            suffix = f" ({'; '.join(metadata)})" if metadata else ""
            lines.append(f"[{item['memory_id']}]{suffix}")
            lines.append(str(item.get("text", "")))
            lines.append("")

    lines.extend(["OUTPUT", "", "Answer the user query directly and concisely."])
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export paired Legacy-vs-C3-v3 evidence prompts for "
            "counterfactual answer evaluation."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--procedures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    memory_records = load_records(args.memories)
    procedure_records = load_records(args.procedures)

    catalog: dict[str, dict[str, Any]] = {}
    for record in [*memory_records, *procedure_records]:
        memory_id = record_id(record)
        if memory_id:
            catalog[memory_id] = record

    rows = [
        row
        for row in read_jsonl(args.predictions)
        if str(row.get("method", "")).lower() == "c3"
    ]
    if not rows:
        raise AssertionError("No C3 rows found.")

    output_rows: list[dict[str, Any]] = []
    missing_ids: set[str] = set()

    for row in rows:
        question_id = str(row.get("question_id", ""))
        shadow = (row.get("debug") or {}).get("c3_v3_arbitration_shadow")

        if not isinstance(shadow, dict):
            raise AssertionError(f"{question_id}: missing C3-v3 shadow trace")

        legacy_ids = [str(x) for x in (shadow.get("legacy_selected_ids") or [])]
        c3_ids = [str(x) for x in (shadow.get("c3_v3_selected_ids") or [])]

        for memory_id in {*legacy_ids, *c3_ids}:
            if memory_id not in catalog:
                missing_ids.add(memory_id)

        trace = trace_index(row)
        legacy_evidence = [
            public_evidence(memory_id, catalog, trace)
            for memory_id in legacy_ids
        ]
        c3_evidence = [
            public_evidence(memory_id, catalog, trace)
            for memory_id in c3_ids
        ]

        query = str(row.get("query", ""))

        output_rows.append(
            {
                "question_id": question_id,
                "question_type": row.get("question_type"),
                "query_mode": row.get("query_mode"),
                "query": query,
                "gold_answer": row.get("gold_answer", ""),
                "should_abstain": bool(row.get("should_abstain", False)),
                "gold_memory_ids": row.get("supporting_memory_ids", []),
                "legacy_selected_ids": legacy_ids,
                "c3_v3_selected_ids": c3_ids,
                "removed_by_c3_v3": shadow.get("removed_by_c3_v3", []),
                "added_by_c3_v3": shadow.get("added_by_c3_v3", []),
                "hard_complete": bool(shadow.get("hard_complete")),
                "legacy_evidence": legacy_evidence,
                "c3_v3_evidence": c3_evidence,
                "legacy_prompt": make_prompt(query=query, evidence=legacy_evidence),
                "c3_v3_prompt": make_prompt(query=query, evidence=c3_evidence),
            }
        )

    if missing_ids:
        raise AssertionError(
            "Selected IDs missing from source catalog: "
            + ", ".join(sorted(missing_ids))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    changed = sum(
        set(row["legacy_selected_ids"]) != set(row["c3_v3_selected_ids"])
        for row in output_rows
    )

    print(
        json.dumps(
            {
                "questions": len(output_rows),
                "changed_evidence_sets": changed,
                "unchanged_evidence_sets": len(output_rows) - changed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
