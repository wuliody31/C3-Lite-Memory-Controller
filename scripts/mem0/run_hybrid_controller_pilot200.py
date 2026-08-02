from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.backbones import MockBackbone
from src.config import load_config
from src.pipeline import C3Pipeline
from src.schemas import (
    MemoryCandidate,
    MemoryType,
    QueryFeatures,
    QueryState,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(
            dict.fromkeys(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        )

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

    return list(
        dict.fromkeys(
            item.strip()
            for item in text.split(";")
            if item.strip()
        )
    )


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def load_memory_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("memories")
        if not isinstance(records, list):
            for key in ("records", "data", "results"):
                if isinstance(payload.get(key), list):
                    records = payload[key]
                    break
    else:
        records = None

    if not isinstance(records, list):
        raise TypeError(f"Unsupported memory JSON structure: {path}")
    if not all(isinstance(item, dict) for item in records):
        raise TypeError("Every memory record must be a JSON object")
    return records


def field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


def build_catalog(
    records: list[dict[str, Any]],
) -> tuple[dict[str, MemoryCandidate], dict[str, list[str]]]:
    catalog: dict[str, MemoryCandidate] = {}
    source_map: dict[str, list[str]] = {}

    for record in records:
        memory_id = str(field(record, "memory_id", "")).strip()
        if not memory_id:
            raise ValueError("Memory record is missing memory_id")
        if memory_id in catalog:
            raise ValueError(f"Duplicate memory_id: {memory_id}")

        memory_type = MemoryType(
            str(field(record, "memory_type", "")).strip()
        )
        source_ids = split_ids(field(record, "source_ids", []))
        metadata = field(record, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        candidate = MemoryCandidate(
            memory_id=memory_id,
            memory_type=memory_type,
            text=str(field(record, "text", "")),
            user_id=str(field(record, "user_id", "")),
            timestamp=parse_datetime(field(record, "timestamp")),
            subject=field(record, "subject"),
            predicate=field(record, "predicate"),
            object_value=field(record, "object_value"),
            status=str(field(record, "status", "current")),
            confidence=float(field(record, "confidence", 1.0)),
            importance=float(field(record, "importance", 0.5)),
            authority=str(field(record, "authority", "unknown")),
            source_ids=source_ids,
            relations=list(field(record, "relations", []) or []),
            metadata=deepcopy(metadata),
        )
        catalog[memory_id] = candidate

        if memory_type == MemoryType.SEMANTIC and source_ids:
            source_map[memory_id] = source_ids
        else:
            source_map[memory_id] = [memory_id]

    return catalog, source_map


def project_ids(
    memory_ids: list[str],
    source_map: dict[str, list[str]],
) -> list[str]:
    output: list[str] = []
    for memory_id in memory_ids:
        output.extend(source_map.get(memory_id, [memory_id]))
    return list(dict.fromkeys(output))


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evidence_metrics(
    predicted_source_ids: list[str],
    gold_source_ids: list[str],
) -> dict[str, Any]:
    predicted = set(predicted_source_ids)
    gold = set(gold_source_ids)
    overlap = predicted & gold

    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold) if gold else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": harmonic_f1(precision, recall),
        "hit": bool(overlap),
        "overlap_source_ids": sorted(overlap),
        "predicted_source_count": len(predicted),
        "gold_source_count": len(gold),
    }


@dataclass(slots=True)
class PoolSpec:
    ordered_ids: list[str]
    source_by_id: dict[str, list[str]]


class FrozenCandidateStore:
    """Replay fixed candidate pools through the unmodified C3 controller."""

    def __init__(
        self,
        *,
        catalog: dict[str, MemoryCandidate],
        pools: dict[str, PoolSpec],
        label: str,
    ) -> None:
        self.catalog = catalog
        self.pools = pools
        self.label = label

    def retrieve(
        self,
        *,
        memory_type: MemoryType,
        state: QueryState,
        features: QueryFeatures,
        top_k: int,
        include_archived: bool = False,
    ) -> list[MemoryCandidate]:
        del features, top_k, include_archived

        question_id = state.session_id
        if not question_id:
            raise ValueError(
                "Hybrid replay requires QueryState.session_id=question_id"
            )
        if question_id not in self.pools:
            raise KeyError(f"No frozen pool for {question_id}")

        spec = self.pools[question_id]
        output: list[MemoryCandidate] = []

        for memory_id in spec.ordered_ids:
            template = self.catalog.get(memory_id)
            if template is None:
                raise KeyError(
                    f"{self.label}: memory {memory_id!r} is absent from catalog"
                )
            if template.memory_type != memory_type:
                continue
            if template.user_id and template.user_id != state.user_id:
                raise AssertionError(
                    f"{question_id}: candidate user mismatch for {memory_id}"
                )

            candidate = deepcopy(template)
            retrieval_sources = spec.source_by_id.get(memory_id, [])
            candidate.metadata["candidate_provider"] = self.label
            candidate.metadata["retrieval_sources"] = retrieval_sources
            candidate.metadata["retrieved_by_c3"] = "c3" in retrieval_sources
            candidate.metadata["retrieved_by_mem0"] = "mem0" in retrieval_sources
            output.append(candidate)

        return output

    def close(self) -> None:
        return None


def make_pool(
    c3_ids: list[str],
    mem0_ids: list[str] | None,
) -> PoolSpec:
    mem0_ids = mem0_ids or []
    ordered_ids = list(dict.fromkeys([*c3_ids, *mem0_ids]))
    c3_set = set(c3_ids)
    mem0_set = set(mem0_ids)

    source_by_id: dict[str, list[str]] = {}
    for memory_id in ordered_ids:
        sources: list[str] = []
        if memory_id in c3_set:
            sources.append("c3")
        if memory_id in mem0_set:
            sources.append("mem0")
        source_by_id[memory_id] = sources

    return PoolSpec(
        ordered_ids=ordered_ids,
        source_by_id=source_by_id,
    )


def selected_source_counts(
    selected_ids: list[str],
    pool: PoolSpec,
) -> dict[str, int]:
    counts = Counter()
    for memory_id in selected_ids:
        sources = pool.source_by_id.get(memory_id, [])
        if sources == ["c3"]:
            counts["c3_only"] += 1
        elif sources == ["mem0"]:
            counts["mem0_only"] += 1
        elif set(sources) == {"c3", "mem0"}:
            counts["both"] += 1
        else:
            counts["unknown"] += 1
    return dict(counts)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate_metrics(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    metrics = [row[prefix] for row in rows]
    return {
        "questions": len(rows),
        "mean_precision": mean(metrics, "precision"),
        "mean_recall": mean(metrics, "recall"),
        "mean_f1": mean(metrics, "f1"),
        "hit_rate": mean(metrics, "hit"),
        "mean_selected": sum(
            len(row[f"{prefix}_selected_ids"])
            for row in rows
        )
        / len(rows),
    }


def category_breakdown(
    rows: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(row)

    return {
        category: aggregate_metrics(category_rows, prefix)
        for category, category_rows in sorted(grouped.items())
    }


def build_report(summary: dict[str, Any]) -> str:
    frozen = summary["frozen_c3_selected"]
    replay = summary["c3_replay_selected"]
    hybrid = summary["hybrid_selected"]
    comparison = summary["paired_comparison"]
    validation = summary["replay_validation"]

    lines = [
        "# C3 + Mem0 Hybrid Controller Replay — Pilot-200",
        "",
        "## Protocol",
        "",
        "- Fixed question set: corrected LoCoMo Pilot-200.",
        "- Original candidate pool: frozen C3 raw retrieval IDs.",
        "- Added candidate pool: provenance-preserving Mem0 top-20 IDs.",
        "- The C3 route planner, shared ranker, candidate budget, conflict "
        "handling, evidence selector, confidence controller and prompt builder "
        "were reused without modification.",
        "- Mock generation was used because this stage evaluates evidence "
        "selection only.",
        "",
        "## Replay validation",
        "",
        f"- Selected-ID set match against frozen C3: "
        f"{validation['selected_set_match_rate']:.4f}",
        f"- Selected-ID order match against frozen C3: "
        f"{validation['selected_order_match_rate']:.4f}",
        f"- Route-type match against frozen C3: "
        f"{validation['route_type_match_rate']:.4f}",
        "",
        "## Evidence results",
        "",
        "| System | Precision | Recall | F1 | Hit | Mean selected |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Frozen C3 selected | {frozen['mean_precision']:.4f} | "
            f"{frozen['mean_recall']:.4f} | {frozen['mean_f1']:.4f} | "
            f"{frozen['hit_rate']:.4f} | {frozen['mean_selected']:.2f} |"
        ),
        (
            f"| C3 controller replay | {replay['mean_precision']:.4f} | "
            f"{replay['mean_recall']:.4f} | {replay['mean_f1']:.4f} | "
            f"{replay['hit_rate']:.4f} | {replay['mean_selected']:.2f} |"
        ),
        (
            f"| C3 + Mem0 hybrid | {hybrid['mean_precision']:.4f} | "
            f"{hybrid['mean_recall']:.4f} | {hybrid['mean_f1']:.4f} | "
            f"{hybrid['hit_rate']:.4f} | {hybrid['mean_selected']:.2f} |"
        ),
        "",
        "## Paired hit outcomes: hybrid versus C3 replay",
        "",
        f"- Both hit: {comparison['both_hit']}",
        f"- C3 replay only: {comparison['c3_only_hit']}",
        f"- Hybrid only (rescued): {comparison['hybrid_only_hit']}",
        f"- Neither hit: {comparison['neither_hit']}",
        f"- Net rescued questions: {comparison['net_rescues']}",
        "",
        "## Hybrid deltas over C3 replay",
        "",
        f"- Precision: {comparison['delta_precision']:+.4f}",
        f"- Recall: {comparison['delta_recall']:+.4f}",
        f"- F1: {comparison['delta_f1']:+.4f}",
        f"- Hit rate: {comparison['delta_hit_rate']:+.4f}",
        "",
        "## Selection provenance",
        "",
        f"- Selected from C3 only: "
        f"{summary['hybrid_selection_provenance'].get('c3_only', 0)}",
        f"- Selected from Mem0 only: "
        f"{summary['hybrid_selection_provenance'].get('mem0_only', 0)}",
        f"- Selected by both retrievers: "
        f"{summary['hybrid_selection_provenance'].get('both', 0)}",
        "",
        "## Decision gate",
        "",
        (
            "Proceed to Qwen3 generation only when replay validation is high, "
            "hybrid-only rescues exceed C3-only losses, and evidence F1 or "
            "recall improves without an unacceptable precision collapse."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen C3 raw candidates and a C3+Mem0 top-20 union "
            "through the unchanged C3 controller on LoCoMo Pilot-200."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--c3-predictions", type=Path, required=True)
    parser.add_argument("--mem0-top20", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    records = load_memory_records(args.memories)
    catalog, source_map = build_catalog(records)

    questions = read_csv(args.questions)
    questions_by_id = {
        str(row["question_id"]): row
        for row in questions
    }

    c3_rows = [
        row
        for row in read_jsonl(args.c3_predictions)
        if str(row.get("method", "")).lower() in {
            "c3",
            "c3_lite_controller",
        }
    ]
    c3_by_id = {
        str(row["question_id"]): row
        for row in c3_rows
    }

    mem0_rows = read_jsonl(args.mem0_top20)
    mem0_by_id = {
        str(row["question_id"]): row
        for row in mem0_rows
    }

    expected_ids = set(questions_by_id)
    if len(expected_ids) != 200:
        raise AssertionError(
            f"Expected 200 unique questions, found {len(expected_ids)}"
        )
    if set(c3_by_id) != expected_ids:
        raise AssertionError("C3 prediction IDs do not match canonical questions")
    if set(mem0_by_id) != expected_ids:
        raise AssertionError("Mem0 IDs do not match canonical questions")

    replay_pools: dict[str, PoolSpec] = {}
    hybrid_pools: dict[str, PoolSpec] = {}

    missing_catalog_ids: set[str] = set()

    for question_id in sorted(expected_ids):
        c3_ids = split_ids(c3_by_id[question_id].get("raw_retrieved_ids"))
        mem0_ids = split_ids(
            mem0_by_id[question_id].get("retrieved_memory_ids")
        )

        for memory_id in [*c3_ids, *mem0_ids]:
            if memory_id not in catalog:
                missing_catalog_ids.add(memory_id)

        replay_pools[question_id] = make_pool(c3_ids, None)
        hybrid_pools[question_id] = make_pool(c3_ids, mem0_ids)

    if missing_catalog_ids:
        examples = sorted(missing_catalog_ids)[:20]
        raise AssertionError(
            f"{len(missing_catalog_ids)} pool IDs absent from catalog: {examples}"
        )

    replay_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=FrozenCandidateStore(
            catalog=catalog,
            pools=replay_pools,
            label="frozen_c3_raw_replay",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )
    hybrid_pipeline = C3Pipeline(
        config=deepcopy(config),
        memory_store=FrozenCandidateStore(
            catalog=catalog,
            pools=hybrid_pools,
            label="c3_mem0_union",
        ),
        procedure_store=None,
        backbone=MockBackbone(),
        prompt_template=args.prompt_template,
    )

    output_rows: list[dict[str, Any]] = []
    provenance_totals = Counter()

    try:
        for question_id in sorted(expected_ids):
            c3 = c3_by_id[question_id]
            canonical = questions_by_id[question_id]

            query = str(c3["query"])
            user_id = str(c3["user_id"])
            question_type = str(
                c3.get(
                    "question_type",
                    canonical.get("question_type", "unknown"),
                )
            )

            gold_memory_ids = split_ids(
                canonical.get("supporting_memory_ids", "")
            )
            gold_source_ids = project_ids(
                gold_memory_ids,
                source_map,
            )

            state = QueryState(
                query=query,
                user_id=user_id,
                session_id=question_id,
            )

            replay_result = replay_pipeline.answer(state)
            hybrid_result = hybrid_pipeline.answer(state)

            frozen_selected_ids = split_ids(c3.get("selected_ids"))
            replay_selected_ids = list(replay_result.selected_ids)
            hybrid_selected_ids = list(hybrid_result.selected_ids)

            frozen_source_ids = project_ids(
                frozen_selected_ids,
                source_map,
            )
            replay_source_ids = project_ids(
                replay_selected_ids,
                source_map,
            )
            hybrid_source_ids = project_ids(
                hybrid_selected_ids,
                source_map,
            )

            frozen_metrics = evidence_metrics(
                frozen_source_ids,
                gold_source_ids,
            )
            replay_metrics = evidence_metrics(
                replay_source_ids,
                gold_source_ids,
            )
            hybrid_metrics = evidence_metrics(
                hybrid_source_ids,
                gold_source_ids,
            )

            selected_provenance = selected_source_counts(
                hybrid_selected_ids,
                hybrid_pools[question_id],
            )
            provenance_totals.update(selected_provenance)

            output_rows.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "query": query,
                    "user_id": user_id,
                    "gold_memory_ids": gold_memory_ids,
                    "gold_source_ids": gold_source_ids,
                    "frozen_c3_selected_ids": frozen_selected_ids,
                    "c3_replay_raw_ids": replay_pools[
                        question_id
                    ].ordered_ids,
                    "hybrid_raw_ids": hybrid_pools[
                        question_id
                    ].ordered_ids,
                    "mem0_added_ids": [
                        memory_id
                        for memory_id in hybrid_pools[
                            question_id
                        ].ordered_ids
                        if memory_id
                        not in set(
                            replay_pools[question_id].ordered_ids
                        )
                    ],
                    "c3_replay_selected_ids": replay_selected_ids,
                    "hybrid_selected_ids": hybrid_selected_ids,
                    "frozen_c3_selected_source_ids": frozen_source_ids,
                    "c3_replay_selected_source_ids": replay_source_ids,
                    "hybrid_selected_source_ids": hybrid_source_ids,
                    "frozen_c3": frozen_metrics,
                    "c3_replay": replay_metrics,
                    "hybrid": hybrid_metrics,
                    "selected_provenance": selected_provenance,
                    "frozen_route_types": c3.get(
                        "selected_memory_types",
                        [],
                    ),
                    "c3_replay_route_types": (
                        replay_result.selected_memory_types
                    ),
                    "hybrid_route_types": (
                        hybrid_result.selected_memory_types
                    ),
                    "c3_replay_decision": (
                        replay_result.decision.value
                    ),
                    "hybrid_decision": hybrid_result.decision.value,
                    "c3_replay_candidate_count_raw": (
                        replay_result.debug[
                            "candidate_count_raw_retrieved"
                        ]
                    ),
                    "hybrid_candidate_count_raw": (
                        hybrid_result.debug[
                            "candidate_count_raw_retrieved"
                        ]
                    ),
                }
            )
    finally:
        replay_pipeline.close()
        hybrid_pipeline.close()

    replay_set_matches = sum(
        set(row["frozen_c3_selected_ids"])
        == set(row["c3_replay_selected_ids"])
        for row in output_rows
    )
    replay_order_matches = sum(
        row["frozen_c3_selected_ids"]
        == row["c3_replay_selected_ids"]
        for row in output_rows
    )
    route_matches = sum(
        row["frozen_route_types"]
        == row["c3_replay_route_types"]
        for row in output_rows
    )

    both_hit = sum(
        row["c3_replay"]["hit"]
        and row["hybrid"]["hit"]
        for row in output_rows
    )
    c3_only_hit = sum(
        row["c3_replay"]["hit"]
        and not row["hybrid"]["hit"]
        for row in output_rows
    )
    hybrid_only_hit = sum(
        not row["c3_replay"]["hit"]
        and row["hybrid"]["hit"]
        for row in output_rows
    )
    neither_hit = len(output_rows) - (
        both_hit
        + c3_only_hit
        + hybrid_only_hit
    )

    frozen_summary = aggregate_metrics(
        output_rows,
        "frozen_c3",
    )
    replay_summary = aggregate_metrics(
        output_rows,
        "c3_replay",
    )
    hybrid_summary = aggregate_metrics(
        output_rows,
        "hybrid",
    )

    summary = {
        "experiment": "rc8_8b5_hybrid_controller_replay_pilot200",
        "questions": len(output_rows),
        "candidate_protocol": {
            "c3": "frozen C3 raw_retrieved_ids",
            "mem0": "Mem0 source-equivalent top-20 retrieved_memory_ids",
            "fusion": "stable union with memory_id deduplication",
            "controller": "unchanged C3 route/ranker/budget/conflict/selector",
            "generation": "MockBackbone; evidence selection only",
        },
        "replay_validation": {
            "selected_set_matches": replay_set_matches,
            "selected_set_match_rate": (
                replay_set_matches / len(output_rows)
            ),
            "selected_order_matches": replay_order_matches,
            "selected_order_match_rate": (
                replay_order_matches / len(output_rows)
            ),
            "route_type_matches": route_matches,
            "route_type_match_rate": (
                route_matches / len(output_rows)
            ),
        },
        "frozen_c3_selected": frozen_summary,
        "c3_replay_selected": replay_summary,
        "hybrid_selected": hybrid_summary,
        "paired_comparison": {
            "both_hit": both_hit,
            "c3_only_hit": c3_only_hit,
            "hybrid_only_hit": hybrid_only_hit,
            "neither_hit": neither_hit,
            "net_rescues": hybrid_only_hit - c3_only_hit,
            "delta_precision": (
                hybrid_summary["mean_precision"]
                - replay_summary["mean_precision"]
            ),
            "delta_recall": (
                hybrid_summary["mean_recall"]
                - replay_summary["mean_recall"]
            ),
            "delta_f1": (
                hybrid_summary["mean_f1"]
                - replay_summary["mean_f1"]
            ),
            "delta_hit_rate": (
                hybrid_summary["hit_rate"]
                - replay_summary["hit_rate"]
            ),
        },
        "hybrid_selection_provenance": dict(provenance_totals),
        "category_breakdown": {
            "c3_replay": category_breakdown(
                output_rows,
                "c3_replay",
            ),
            "hybrid": category_breakdown(
                output_rows,
                "hybrid",
            ),
        },
        "mean_candidate_counts": {
            "c3_replay_raw": sum(
                row["c3_replay_candidate_count_raw"]
                for row in output_rows
            )
            / len(output_rows),
            "hybrid_raw": sum(
                row["hybrid_candidate_count_raw"]
                for row in output_rows
            )
            / len(output_rows),
            "mem0_unique_added": sum(
                len(row["mem0_added_ids"])
                for row in output_rows
            )
            / len(output_rows),
        },
    }

    write_jsonl(
        args.output_dir / "hybrid_controller_replay.jsonl",
        output_rows,
    )
    (
        args.output_dir / "hybrid_controller_summary.json"
    ).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (
        args.output_dir / "hybrid_controller_report.md"
    ).write_text(
        build_report(summary),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("C3 + MEM0 HYBRID CONTROLLER REPLAY: PASSED")


if __name__ == "__main__":
    main()
