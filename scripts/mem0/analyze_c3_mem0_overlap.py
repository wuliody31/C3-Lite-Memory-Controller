from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


C3_METHODS = {"c3", "c3_lite_controller"}
C3_STAGES = ("selected", "ranked10", "raw30")
MEM0_CONFIGS = ("top5", "top20")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def load_memories(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
        return payload["memories"]
    raise TypeError(f"Unsupported memory JSON structure: {path}")


def get_field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(values))
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
    return list(dict.fromkeys(part.strip() for part in text.split(";") if part.strip()))


def build_source_map(memories: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for memory in memories:
        memory_id = str(get_field(memory, "memory_id", "")).strip()
        if not memory_id:
            raise ValueError("Memory without memory_id")
        memory_type = str(get_field(memory, "memory_type", "")).strip()
        source_ids = split_ids(get_field(memory, "source_ids", []))
        if memory_type == "semantic" and source_ids:
            mapping[memory_id] = source_ids
        else:
            mapping[memory_id] = [memory_id]
    return mapping


def project_ids(ids: list[str], source_map: dict[str, list[str]]) -> list[str]:
    projected: list[str] = []
    for memory_id in ids:
        projected.extend(source_map.get(memory_id, [memory_id]))
    return list(dict.fromkeys(projected))


def selected_sources(
    row: dict[str, Any],
    source_map: dict[str, list[str]],
) -> list[str]:
    """Project C3 selected evidence with adapter-v02 annotations.

    The frozen Pilot-200 prediction may contain source annotations
    created before the LoCoMo gold-evidence repair. The canonical
    adapter-v02 source map therefore takes precedence whenever a
    memory ID is available.
    """
    evidence = row.get("selected_evidence")
    projected: list[str] = []

    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue

            memory_id = str(
                item.get("memory_id", "")
            ).strip()

            if memory_id:
                projected.extend(
                    source_map.get(
                        memory_id,
                        [memory_id],
                    )
                )
                continue

            projected.extend(
                split_ids(
                    item.get("source_ids")
                )
            )

    if projected:
        return list(
            dict.fromkeys(projected)
        )

    return project_ids(
        split_ids(row.get("selected_ids")),
        source_map,
    )


def stage_sources(
    row: dict[str, Any],
    stage: str,
    source_map: dict[str, list[str]],
) -> list[str]:
    if stage == "selected":
        return selected_sources(row, source_map)
    if stage == "ranked10":
        return project_ids(split_ids(row.get("ranked_candidate_ids")), source_map)
    if stage == "raw30":
        return project_ids(split_ids(row.get("raw_retrieved_ids")), source_map)
    raise ValueError(f"Unknown C3 stage: {stage}")


def score(candidate_ids: list[str], gold_ids: list[str]) -> dict[str, Any]:
    candidate_set = set(candidate_ids)
    gold_set = set(gold_ids)
    overlap = candidate_set & gold_set
    precision = len(overlap) / len(candidate_set) if candidate_set else 0.0
    recall = len(overlap) / len(gold_set) if gold_set else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "overlap_ids": sorted(overlap),
        "hit": bool(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_method(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows]
    return {
        "questions": len(values),
        "hit_rate": sum(int(value["hit"]) for value in values) / len(values),
        "mean_precision": sum(value["precision"] for value in values) / len(values),
        "mean_recall": sum(value["recall"] for value in values) / len(values),
        "mean_f1": sum(value["f1"] for value in values) / len(values),
        "mean_candidate_count": sum(value["candidate_count"] for value in values) / len(values),
    }


def summarize_pair(
    rows: list[dict[str, Any]],
    c3_key: str,
    mem0_key: str,
    union_key: str,
) -> dict[str, Any]:
    both = c3_only = mem0_only = neither = 0
    incremental_gold_ids = 0
    for row in rows:
        c3 = row[c3_key]
        mem0 = row[mem0_key]
        if c3["hit"] and mem0["hit"]:
            both += 1
        elif c3["hit"]:
            c3_only += 1
        elif mem0["hit"]:
            mem0_only += 1
        else:
            neither += 1
        incremental_gold_ids += len(
            set(mem0["overlap_ids"]) - set(c3["overlap_ids"])
        )

    c3_summary = summarize_method(rows, c3_key)
    mem0_summary = summarize_method(rows, mem0_key)
    union_summary = summarize_method(rows, union_key)

    return {
        "questions": len(rows),
        "both_hit": both,
        "c3_only_hit": c3_only,
        "mem0_only_hit": mem0_only,
        "neither_hit": neither,
        "c3": c3_summary,
        "mem0": mem0_summary,
        "union": union_summary,
        "delta_hit_rate_union_minus_c3": (
            union_summary["hit_rate"] - c3_summary["hit_rate"]
        ),
        "delta_recall_union_minus_c3": (
            union_summary["mean_recall"] - c3_summary["mean_recall"]
        ),
        "incremental_gold_source_ids_from_mem0": incremental_gold_ids,
        "mean_incremental_gold_source_ids_per_question": (
            incremental_gold_ids / len(rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse C3–Mem0 evidence overlap and union recall on the "
            "aligned LoCoMo Pilot-200 question set."
        )
    )
    parser.add_argument("--c3-jsonl", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--mem0-top5", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_map = build_source_map(
        load_memories(args.memories)
    )

    questions = read_csv(args.questions)

    questions_by_id = {
        str(row["question_id"]): row
        for row in questions
    }

    if (
        len(questions) != 200
        or len(questions_by_id) != 200
    ):
        raise AssertionError(
            "Expected exactly 200 unique "
            "canonical Pilot questions."
        )

    c3_all = read_jsonl(args.c3_jsonl)
    c3_rows = [
        row
        for row in c3_all
        if str(row.get("method", "")).lower() in C3_METHODS
    ]
    mem0_rows = {
        "top5": read_jsonl(args.mem0_top5),
        "top20": read_jsonl(args.mem0_top20),
    }

    c3_by_id = {str(row["question_id"]): row for row in c3_rows}
    mem0_by_config = {
        name: {str(row["question_id"]): row for row in rows}
        for name, rows in mem0_rows.items()
    }

    if len(c3_rows) != 200 or len(c3_by_id) != 200:
        raise AssertionError("Expected exactly 200 unique C3 rows")

    question_ids = set(c3_by_id)

    if set(questions_by_id) != question_ids:
        raise AssertionError(
            "Canonical question CSV does not "
            "match the C3 Pilot-200 IDs."
        )

    for name, mapping in mem0_by_config.items():
        if len(mapping) != 200 or set(mapping) != question_ids:
            raise AssertionError(f"Question mismatch for Mem0 {name}")

    per_question: list[dict[str, Any]] = []

    for question_id in sorted(question_ids):
        c3 = c3_by_id[question_id]
        mem0_top5 = mem0_by_config["top5"][question_id]
        mem0_top20 = mem0_by_config["top20"][question_id]

        canonical_question = (
            questions_by_id[question_id]
        )

        gold_ids = project_ids(
            split_ids(
                canonical_question.get(
                    "supporting_memory_ids",
                    "",
                )
            ),
            source_map,
        )

        mem0_gold_ids = split_ids(
            mem0_top20.get(
                "gold_source_ids",
                mem0_top20.get(
                    "gold_ids",
                ),
            )
        )

        if set(gold_ids) != set(mem0_gold_ids):
            raise AssertionError(
                "Canonical gold mismatch for "
                f"{question_id}: "
                f"questions={gold_ids}, "
                f"Mem0={mem0_gold_ids}"
            )

        row: dict[str, Any] = {
            "question_id": question_id,
            "user_id": c3.get("user_id"),
            "question_type": c3.get("question_type"),
            "query": c3.get("query"),
            "gold_source_ids": gold_ids,
        }

        for stage in C3_STAGES:
            key = f"c3_{stage}"
            row[key] = score(
                stage_sources(c3, stage, source_map),
                gold_ids,
            )

        for config, mem0_row in (("top5", mem0_top5), ("top20", mem0_top20)):
            key = f"mem0_{config}"
            mem0_ids = split_ids(
                mem0_row.get(
                    "retrieved_source_ids",
                    mem0_row.get("projected_source_ids"),
                )
            )
            row[key] = score(mem0_ids, gold_ids)

        for stage in C3_STAGES:
            for config in MEM0_CONFIGS:
                union_ids = list(
                    dict.fromkeys(
                        row[f"c3_{stage}"]["candidate_ids"]
                        + row[f"mem0_{config}"]["candidate_ids"]
                    )
                )
                row[f"union_{stage}_{config}"] = score(union_ids, gold_ids)

        per_question.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = args.output_dir / "c3_mem0_overlap_per_question.jsonl"
    with per_question_path.open("w", encoding="utf-8") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "questions": len(per_question),
        "comparisons": {},
    }

    for stage in C3_STAGES:
        for config in MEM0_CONFIGS:
            name = f"c3_{stage}_vs_mem0_{config}"
            summary["comparisons"][name] = summarize_pair(
                per_question,
                f"c3_{stage}",
                f"mem0_{config}",
                f"union_{stage}_{config}",
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        grouped[str(row["question_type"])].append(row)

    category_summary: dict[str, Any] = {}
    for category, rows in sorted(grouped.items()):
        category_summary[category] = {}
        for stage in C3_STAGES:
            for config in MEM0_CONFIGS:
                name = f"c3_{stage}_vs_mem0_{config}"
                category_summary[category][name] = summarize_pair(
                    rows,
                    f"c3_{stage}",
                    f"mem0_{config}",
                    f"union_{stage}_{config}",
                )

    (args.output_dir / "c3_mem0_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "c3_mem0_overlap_by_category.json").write_text(
        json.dumps(category_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# C3–Mem0 Pilot-200 Overlap and Union Analysis",
        "",
        "| Comparison | Both | C3 only | Mem0 only | Neither | C3 hit | Mem0 hit | Union hit | Δ hit | C3 recall | Mem0 recall | Union recall | Δ recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["comparisons"].items():
        report_lines.append(
            "| {name} | {both} | {c3_only} | {mem0_only} | {neither} | "
            "{c3_hit:.4f} | {mem0_hit:.4f} | {union_hit:.4f} | {delta_hit:+.4f} | "
            "{c3_recall:.4f} | {mem0_recall:.4f} | {union_recall:.4f} | {delta_recall:+.4f} |".format(
                name=name,
                both=result["both_hit"],
                c3_only=result["c3_only_hit"],
                mem0_only=result["mem0_only_hit"],
                neither=result["neither_hit"],
                c3_hit=result["c3"]["hit_rate"],
                mem0_hit=result["mem0"]["hit_rate"],
                union_hit=result["union"]["hit_rate"],
                delta_hit=result["delta_hit_rate_union_minus_c3"],
                c3_recall=result["c3"]["mean_recall"],
                mem0_recall=result["mem0"]["mean_recall"],
                union_recall=result["union"]["mean_recall"],
                delta_recall=result["delta_recall_union_minus_c3"],
            )
        )

    (args.output_dir / "c3_mem0_overlap_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("C3–MEM0 OVERLAP/UNION ANALYSIS: PASSED")


if __name__ == "__main__":
    main()
