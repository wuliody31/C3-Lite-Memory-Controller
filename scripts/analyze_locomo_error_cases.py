#!/usr/bin/env python3
"""
LoCoMo 1786 post-hoc error analysis for the frozen C3-v3 external evaluation.

This script is ANALYSIS-ONLY:
- it does not modify C3,
- it does not regenerate answers,
- it does not change retrieval outputs,
- it only reads frozen artifacts and produces deterministic diagnostic reports.

Primary comparison:
    c3_v3 vs mem0_c3_matched
Secondary comparisons:
    c3_v3 vs simple_retrieval
    c3_v3 vs mem0_top20

The automatic taxonomy is a CANDIDATE taxonomy for manual paper audit.
It must not be treated as a substitute for semantic human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIANTS = (
    "simple_retrieval",
    "mem0_top20",
    "mem0_c3_matched",
    "c3_v3",
)

PRIMARY_BASELINE = "mem0_c3_matched"

# These are descriptive heuristics for selecting cases for manual review.
# They are NOT statistical significance thresholds.
ANSWER_MATERIAL = 0.05
ANSWER_SMALL = 0.02
EVIDENCE_MATERIAL = 0.03

TIME_PATTERNS = re.compile(
    r"\b("
    r"when|what date|which date|what day|which day|what year|which year|"
    r"what month|which month|how long|how many days|how many weeks|"
    r"how many months|how many years|before|after|first|last|recently|"
    r"earlier|later|timeline|over time"
    r")\b",
    re.I,
)
CURRENT_PATTERNS = re.compile(
    r"\b(current|currently|now|latest|still|today|present|at the moment)\b",
    re.I,
)
MULTIFACET_PATTERNS = re.compile(
    r"\b("
    r"all|both|several|multiple|different|various|activities|things|"
    r"ways|examples|events|places|people|interests|hobbies|"
    r"what .* and |which .* and "
    r")\b",
    re.I,
)
CAUSAL_PATTERNS = re.compile(
    r"\b("
    r"why|reason|because|cause|caused|lead to|led to|result|impact|"
    r"affect|affected|relationship|connection|change|changed|"
    r"how did|how has|how have"
    r")\b",
    re.I,
)
LOCATION_PATTERNS = re.compile(
    r"\b(where|location|place|city|country|live|lives|living|moved|move)\b",
    re.I,
)
PERSON_PATTERNS = re.compile(
    r"\b(who|whose|person|friend|mother|father|sister|brother|partner)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(x) for x in value)
    return str(value or "")


def lexical_intent_tags(query: str, gold_answer: str) -> list[str]:
    text = f"{query} {gold_answer}".strip()
    tags: list[str] = []
    if TIME_PATTERNS.search(text):
        tags.append("temporal_or_exact_detail")
    if CURRENT_PATTERNS.search(text):
        tags.append("current_state")
    if MULTIFACET_PATTERNS.search(text) or query.lower().count(" and ") >= 1:
        tags.append("multi_facet")
    if CAUSAL_PATTERNS.search(text):
        tags.append("causal_or_relational")
    if LOCATION_PATTERNS.search(text):
        tags.append("location")
    if PERSON_PATTERNS.search(text):
        tags.append("person_or_relation")
    if not tags:
        tags.append("other")
    return tags


def extract_evidence_blocks(prompt: str) -> list[dict[str, str]]:
    """
    Parse the neutral prompt emitted by the frozen LoCoMo harness.
    Returns exact evidence text used for generation when available.
    """
    if not prompt:
        return []

    marker = "MEMORY EVIDENCE"
    end_marker = "OUTPUT REQUIREMENTS"

    if marker not in prompt:
        return []

    section = prompt.split(marker, 1)[1]
    if end_marker in section:
        section = section.split(end_marker, 1)[0]

    section = section.strip()
    if not section or section == "- None":
        return []

    pattern = re.compile(
        r"^\[([^\]]+)\]\s+\(time=([^)]+)\)\s*$\n"
        r"(.*?)(?=^\[[^\]]+\]\s+\(time=[^)]+\)\s*$|\Z)",
        re.M | re.S,
    )

    blocks: list[dict[str, str]] = []
    for match in pattern.finditer(section):
        text = match.group(3).strip()
        blocks.append(
            {
                "memory_id": match.group(1).strip(),
                "time": match.group(2).strip(),
                "text": text,
            }
        )
    return blocks


def compact_text(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def compare_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_overlap = set(str(x) for x in left.get("evidence_overlap_source_ids", []))
    right_overlap = set(str(x) for x in right.get("evidence_overlap_source_ids", []))

    return {
        "answer_f1_delta": as_float(right.get("answer_f1")) - as_float(left.get("answer_f1")),
        "evidence_precision_delta": (
            as_float(right.get("evidence_precision"))
            - as_float(left.get("evidence_precision"))
        ),
        "evidence_recall_delta": (
            as_float(right.get("evidence_recall"))
            - as_float(left.get("evidence_recall"))
        ),
        "evidence_f1_delta": (
            as_float(right.get("evidence_f1"))
            - as_float(left.get("evidence_f1"))
        ),
        "selected_count_delta": int(right.get("selected_count", 0) or 0)
        - int(left.get("selected_count", 0) or 0),
        "input_tokens_delta": as_float(right.get("input_tokens_numeric"))
        - as_float(left.get("input_tokens_numeric")),
        "latency_ms_delta": as_float(right.get("generation_latency_ms"))
        - as_float(left.get("generation_latency_ms")),
        "left_hit": as_bool(left.get("evidence_hit")),
        "right_hit": as_bool(right.get("evidence_hit")),
        "lost_annotated_source_ids": sorted(left_overlap - right_overlap),
        "gained_annotated_source_ids": sorted(right_overlap - left_overlap),
        "same_annotated_overlap": left_overlap == right_overlap,
    }


def auto_tags(
    item: dict[str, Any],
    matched: dict[str, Any],
    simple: dict[str, Any],
    top20: dict[str, Any],
) -> list[str]:
    tags: list[str] = list(item["lexical_intent_tags"])

    ad = matched["answer_f1_delta"]
    ed = matched["evidence_f1_delta"]

    if ed >= EVIDENCE_MATERIAL and ad <= -ANSWER_SMALL:
        tags.append("evidence_gain_but_answer_loss")
    if ed >= EVIDENCE_MATERIAL and abs(ad) < ANSWER_SMALL:
        tags.append("evidence_gain_answer_near_tie")
    if ed >= EVIDENCE_MATERIAL and ad >= ANSWER_MATERIAL:
        tags.append("evidence_and_answer_gain")
    if ed <= -EVIDENCE_MATERIAL and ad <= -ANSWER_MATERIAL:
        tags.append("evidence_and_answer_loss")

    if matched["left_hit"] and not matched["right_hit"]:
        tags.append("annotated_support_hit_loss")
    if not matched["left_hit"] and matched["right_hit"]:
        tags.append("annotated_support_hit_gain")

    if matched["lost_annotated_source_ids"]:
        tags.append("annotated_support_removed_vs_matched")
    if matched["gained_annotated_source_ids"]:
        tags.append("annotated_support_gained_vs_matched")

    if matched["same_annotated_overlap"] and ad <= -ANSWER_MATERIAL:
        tags.append("same_gold_overlap_but_answer_loss")
    if (
        matched["right_hit"]
        and ad <= -ANSWER_MATERIAL
        and matched["same_annotated_overlap"]
    ):
        tags.append("hit_preserved_but_answer_loss")

    if simple["answer_f1_delta"] <= -ANSWER_MATERIAL:
        tags.append("answer_loss_vs_simple")
    if top20["answer_f1_delta"] <= -ANSWER_MATERIAL:
        tags.append("answer_loss_vs_mem0_top20")
    if simple["answer_f1_delta"] >= ANSWER_MATERIAL:
        tags.append("answer_gain_vs_simple")
    if top20["answer_f1_delta"] >= ANSWER_MATERIAL:
        tags.append("answer_gain_vs_mem0_top20")

    c3 = item["variants"]["c3_v3"]
    matched_row = item["variants"]["mem0_c3_matched"]

    if (
        as_float(c3.get("evidence_recall")) > as_float(matched_row.get("evidence_recall"))
        and as_float(c3.get("answer_f1")) < as_float(matched_row.get("answer_f1"))
    ):
        tags.append("higher_recall_lower_answer_vs_matched")

    if (
        as_float(c3.get("evidence_precision")) > as_float(matched_row.get("evidence_precision"))
        and as_float(c3.get("answer_f1")) < as_float(matched_row.get("answer_f1"))
    ):
        tags.append("higher_precision_lower_answer_vs_matched")

    # Candidate only. Semantic equivalence must be manually judged.
    if (
        0.0 < abs(ad) <= 0.08
        and item["variants"]["c3_v3"].get("answer", "").strip()
        != item["variants"]["mem0_c3_matched"].get("answer", "").strip()
    ):
        tags.append("manual_semantic_metric_audit_candidate")

    return sorted(set(tags))


def paper_importance_score(
    item: dict[str, Any],
    matched: dict[str, Any],
    simple: dict[str, Any],
    top20: dict[str, Any],
) -> float:
    """
    Descriptive ranking only. Designed to surface cases that explain the paper:
    - evidence/answer decoupling,
    - annotated support loss,
    - compression boundary,
    - strong C3 successes.
    """
    score = 0.0

    ad = matched["answer_f1_delta"]
    ed = matched["evidence_f1_delta"]

    # Primary matched-budget evidence-answer decoupling.
    if ed > 0 and ad < 0:
        score += 8.0 * min(ed, 1.0) + 8.0 * min(-ad, 1.0)
    if ed > 0 and ad > 0:
        score += 5.0 * min(ed, 1.0) + 5.0 * min(ad, 1.0)

    # Annotated source changes matter for mechanistic inspection.
    score += 1.5 * len(matched["lost_annotated_source_ids"])
    score += 1.0 * len(matched["gained_annotated_source_ids"])

    if matched["left_hit"] and not matched["right_hit"]:
        score += 3.0
    if matched["same_annotated_overlap"] and ad <= -ANSWER_MATERIAL:
        score += 2.0

    # External compression boundary vs larger-context retrieval.
    score += 3.0 * max(0.0, -simple["answer_f1_delta"])
    score += 2.0 * max(0.0, -top20["answer_f1_delta"])

    # Prefer interpretable temporal/multi-facet/relational cases.
    tags = set(item["lexical_intent_tags"])
    if "temporal_or_exact_detail" in tags:
        score += 0.6
    if "current_state" in tags:
        score += 0.6
    if "multi_facet" in tags:
        score += 0.6
    if "causal_or_relational" in tags:
        score += 0.6

    # Strong C3 success cases are useful counterexamples.
    if ad >= ANSWER_MATERIAL:
        score += 2.0 * ad

    return score


def diagnostic_row(
    qid: str,
    input_row: dict[str, Any],
    variant_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}

    for variant in VARIANTS:
        row = variant_rows[variant]
        variants[variant] = {
            "answer": str(row.get("answer", "")),
            "answer_f1": as_float(row.get("answer_f1")),
            "answer_exact_match": as_float(row.get("answer_exact_match")),
            "evidence_precision": as_float(row.get("evidence_precision")),
            "evidence_recall": as_float(row.get("evidence_recall")),
            "evidence_f1": as_float(row.get("evidence_f1")),
            "evidence_hit": as_bool(row.get("evidence_hit")),
            "selected_count": int(row.get("selected_count", 0) or 0),
            "selected_ids": [str(x) for x in row.get("selected_ids", [])],
            "selected_source_ids": [
                str(x) for x in row.get("selected_source_ids", [])
            ],
            "evidence_overlap_source_ids": [
                str(x) for x in row.get("evidence_overlap_source_ids", [])
            ],
            "input_tokens_numeric": as_float(row.get("input_tokens_numeric")),
            "output_tokens_numeric": as_float(row.get("output_tokens_numeric")),
            "generation_latency_ms": as_float(row.get("generation_latency_ms")),
            "evidence_blocks": extract_evidence_blocks(
                str(row.get("final_prompt", ""))
            ),
        }

    item = {
        "question_id": qid,
        "user_id": str(input_row.get("user_id", "")),
        "question_type": str(input_row.get("question_type", "unknown")),
        "query": str(input_row.get("query", "")),
        "gold_answer": normalize_answer(input_row.get("gold_answer", "")),
        "should_abstain": as_bool(input_row.get("should_abstain")),
        "gold_memory_ids": [str(x) for x in input_row.get("gold_memory_ids", [])],
        "gold_source_ids": [str(x) for x in input_row.get("gold_source_ids", [])],
        "lexical_intent_tags": lexical_intent_tags(
            str(input_row.get("query", "")),
            normalize_answer(input_row.get("gold_answer", "")),
        ),
        "variants": variants,
    }

    comparisons = {
        "c3_minus_mem0_c3_matched": compare_row(
            variant_rows["mem0_c3_matched"],
            variant_rows["c3_v3"],
        ),
        "c3_minus_simple": compare_row(
            variant_rows["simple_retrieval"],
            variant_rows["c3_v3"],
        ),
        "c3_minus_mem0_top20": compare_row(
            variant_rows["mem0_top20"],
            variant_rows["c3_v3"],
        ),
    }
    item["comparisons"] = comparisons

    matched = comparisons["c3_minus_mem0_c3_matched"]
    simple = comparisons["c3_minus_simple"]
    top20 = comparisons["c3_minus_mem0_top20"]

    item["auto_tags"] = auto_tags(item, matched, simple, top20)
    item["paper_importance_score"] = paper_importance_score(
        item, matched, simple, top20
    )
    return item


def bucket_name(item: dict[str, Any]) -> str:
    tags = set(item["auto_tags"])
    matched = item["comparisons"]["c3_minus_mem0_c3_matched"]

    if "evidence_gain_but_answer_loss" in tags:
        return "A_evidence_answer_decoupling"
    if "annotated_support_removed_vs_matched" in tags and matched["answer_f1_delta"] < 0:
        return "B_annotated_support_loss"
    if matched["answer_f1_delta"] >= ANSWER_MATERIAL:
        return "C_c3_matched_success"
    if (
        "answer_loss_vs_simple" in tags
        or "answer_loss_vs_mem0_top20" in tags
    ):
        return "D_large_context_compression_boundary"
    if "temporal_or_exact_detail" in tags or "current_state" in tags:
        return "E_temporal_currentness"
    if "multi_facet" in tags or "causal_or_relational" in tags:
        return "F_multifacet_relational"
    if "manual_semantic_metric_audit_candidate" in tags:
        return "G_metric_audit_candidate"
    return "H_other"


def select_balanced_cases(
    items: list[dict[str, Any]],
    top_cases: int,
) -> list[dict[str, Any]]:
    """
    Deterministic, diversity-aware selection.
    Up to 30 cases are selected from paper-relevant buckets.
    """
    quotas = [
        ("A_evidence_answer_decoupling", 8),
        ("B_annotated_support_loss", 5),
        ("C_c3_matched_success", 5),
        ("D_large_context_compression_boundary", 4),
        ("E_temporal_currentness", 3),
        ("F_multifacet_relational", 3),
        ("G_metric_audit_candidate", 2),
    ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[bucket_name(item)].append(item)

    for bucket in grouped:
        grouped[bucket].sort(
            key=lambda x: (
                -as_float(x.get("paper_importance_score")),
                str(x["question_id"]),
            )
        )

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for bucket, quota in quotas:
        for item in grouped.get(bucket, []):
            if len(selected) >= top_cases or quota <= 0:
                break
            qid = str(item["question_id"])
            if qid in seen:
                continue
            copy = dict(item)
            copy["selection_bucket"] = bucket
            selected.append(copy)
            seen.add(qid)
            quota -= 1

    if len(selected) < top_cases:
        remaining = sorted(
            (x for x in items if str(x["question_id"]) not in seen),
            key=lambda x: (
                -as_float(x.get("paper_importance_score")),
                str(x["question_id"]),
            ),
        )
        for item in remaining:
            if len(selected) >= top_cases:
                break
            copy = dict(item)
            copy["selection_bucket"] = bucket_name(item)
            selected.append(copy)
            seen.add(str(item["question_id"]))

    for rank, item in enumerate(selected, 1):
        item["paper_case_rank"] = rank

    return selected


def delta_sign(value: float, eps: float = 1e-12) -> str:
    if value > eps:
        return "gain"
    if value < -eps:
        return "loss"
    return "tie"


def practical_delta_band(value: float) -> str:
    if value >= ANSWER_SMALL:
        return "gain_ge_0.02"
    if value <= -ANSWER_SMALL:
        return "loss_le_-0.02"
    return "within_0.02"


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    matched_matrix = Counter()
    practical_matrix = Counter()
    hit_transitions = Counter()
    tags = Counter()
    question_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    intent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        matched = item["comparisons"]["c3_minus_mem0_c3_matched"]
        matched_matrix[
            (
                delta_sign(matched["evidence_f1_delta"]),
                delta_sign(matched["answer_f1_delta"]),
            )
        ] += 1

        practical_matrix[
            (
                "evidence_gain_ge_0.03"
                if matched["evidence_f1_delta"] >= EVIDENCE_MATERIAL
                else (
                    "evidence_loss_le_-0.03"
                    if matched["evidence_f1_delta"] <= -EVIDENCE_MATERIAL
                    else "evidence_within_0.03"
                ),
                practical_delta_band(matched["answer_f1_delta"]),
            )
        ] += 1

        hit_transitions[
            (
                "hit" if matched["left_hit"] else "miss",
                "hit" if matched["right_hit"] else "miss",
            )
        ] += 1

        tags.update(item["auto_tags"])
        question_types[item["question_type"]].append(item)
        for tag in item["lexical_intent_tags"]:
            intent_groups[tag].append(item)

    def subgroup(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "mean_delta_answer_f1_c3_minus_matched": mean(
                x["comparisons"]["c3_minus_mem0_c3_matched"]["answer_f1_delta"]
                for x in rows
            ),
            "mean_delta_evidence_f1_c3_minus_matched": mean(
                x["comparisons"]["c3_minus_mem0_c3_matched"]["evidence_f1_delta"]
                for x in rows
            ),
            "mean_delta_answer_f1_c3_minus_simple": mean(
                x["comparisons"]["c3_minus_simple"]["answer_f1_delta"]
                for x in rows
            ),
            "mean_delta_answer_f1_c3_minus_mem0_top20": mean(
                x["comparisons"]["c3_minus_mem0_top20"]["answer_f1_delta"]
                for x in rows
            ),
        }

    return {
        "questions": len(items),
        "primary_comparison": "c3_v3_minus_mem0_c3_matched",
        "heuristic_thresholds_for_case_selection_only": {
            "answer_material": ANSWER_MATERIAL,
            "answer_small_band": ANSWER_SMALL,
            "evidence_material": EVIDENCE_MATERIAL,
        },
        "matched_exact_sign_matrix": {
            f"evidence_{evidence}__answer_{answer}": count
            for (evidence, answer), count in sorted(matched_matrix.items())
        },
        "matched_descriptive_threshold_matrix": {
            f"{evidence}__{answer}": count
            for (evidence, answer), count in sorted(practical_matrix.items())
        },
        "matched_evidence_hit_transitions": {
            f"mem0_{left}_to_c3_{right}": count
            for (left, right), count in sorted(hit_transitions.items())
        },
        "auto_tag_counts": dict(tags.most_common()),
        "by_question_type": {
            key: subgroup(rows)
            for key, rows in sorted(question_types.items())
        },
        "by_lexical_intent": {
            key: subgroup(rows)
            for key, rows in sorted(intent_groups.items())
        },
    }


def variant_summary_line(v: dict[str, Any]) -> str:
    return (
        f"AnswerF1={v['answer_f1']:.4f}, "
        f"EvidenceF1={v['evidence_f1']:.4f}, "
        f"P={v['evidence_precision']:.4f}, "
        f"R={v['evidence_recall']:.4f}, "
        f"hit={v['evidence_hit']}, "
        f"n={v['selected_count']}, "
        f"input={v['input_tokens_numeric']:.0f}"
    )


def evidence_markdown(v: dict[str, Any], max_blocks: int = 5) -> str:
    blocks = v.get("evidence_blocks", [])
    if not blocks:
        return "_Evidence text unavailable in parsed prompt._"

    lines: list[str] = []
    for block in blocks[:max_blocks]:
        lines.append(
            f"- `{block['memory_id']}` @ `{block['time']}` — "
            f"{compact_text(block['text'], 520)}"
        )
    if len(blocks) > max_blocks:
        lines.append(f"- … {len(blocks) - max_blocks} more evidence blocks omitted")
    return "\n".join(lines)


def write_selected_markdown(
    path: Path,
    selected: list[dict[str, Any]],
) -> None:
    lines = [
        "# LoCoMo 1786 — Selected Error-Analysis Cases",
        "",
        "Automatic labels below are **candidate labels for manual audit**, "
        "not final semantic judgments.",
        "",
        "Primary comparison: **C3-v3 vs Mem0 evidence-count matched**.",
        "",
    ]

    for item in selected:
        m = item["comparisons"]["c3_minus_mem0_c3_matched"]
        s = item["comparisons"]["c3_minus_simple"]
        t = item["comparisons"]["c3_minus_mem0_top20"]
        c3 = item["variants"]["c3_v3"]
        matched = item["variants"]["mem0_c3_matched"]

        lines.extend(
            [
                f"## {item['paper_case_rank']}. {item['question_id']}",
                "",
                f"**Selection bucket:** `{item['selection_bucket']}`  ",
                f"**Question type:** `{item['question_type']}`  ",
                f"**Auto tags:** {', '.join(f'`{x}`' for x in item['auto_tags'])}",
                "",
                f"**Question:** {item['query']}",
                "",
                f"**Gold answer:** {item['gold_answer']}",
                "",
                "### Matched-budget comparison",
                "",
                f"- Mem0 matched: {variant_summary_line(matched)}",
                f"- C3-v3: {variant_summary_line(c3)}",
                f"- Δ Answer F1: **{m['answer_f1_delta']:+.4f}**",
                f"- Δ Evidence F1: **{m['evidence_f1_delta']:+.4f}**",
                f"- Lost annotated source IDs: `{m['lost_annotated_source_ids']}`",
                f"- Gained annotated source IDs: `{m['gained_annotated_source_ids']}`",
                "",
                "**Mem0 matched answer:**",
                "",
                f"> {compact_text(matched['answer'], 900)}",
                "",
                "**C3 answer:**",
                "",
                f"> {compact_text(c3['answer'], 900)}",
                "",
                "**Mem0 matched evidence:**",
                "",
                evidence_markdown(matched),
                "",
                "**C3 evidence:**",
                "",
                evidence_markdown(c3),
                "",
                "### Larger-context reference deltas",
                "",
                f"- C3 − Simple Answer F1: `{s['answer_f1_delta']:+.4f}`",
                f"- C3 − Mem0@20 Answer F1: `{t['answer_f1_delta']:+.4f}`",
                "",
                "### Manual audit",
                "",
                "- Final taxonomy: ",
                "- Semantic-equivalence / metric artifact? ",
                "- Did C3 remove annotated support? ",
                "- Did it remove answer-useful but non-annotated context? ",
                "- Is this temporal/currentness, multi-facet, relational/causal, or other? ",
                "- Paper decision: main text / appendix / exclude",
                "",
                "---",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_selected_csv(path: Path, selected: list[dict[str, Any]]) -> None:
    fields = [
        "paper_case_rank",
        "question_id",
        "user_id",
        "question_type",
        "selection_bucket",
        "query",
        "gold_answer",
        "auto_tags",
        "c3_answer_f1",
        "matched_answer_f1",
        "delta_answer_f1_c3_minus_matched",
        "c3_evidence_f1",
        "matched_evidence_f1",
        "delta_evidence_f1_c3_minus_matched",
        "c3_evidence_precision",
        "matched_evidence_precision",
        "c3_evidence_recall",
        "matched_evidence_recall",
        "c3_hit",
        "matched_hit",
        "lost_annotated_source_ids",
        "gained_annotated_source_ids",
        "delta_answer_f1_c3_minus_simple",
        "delta_answer_f1_c3_minus_mem0_top20",
        "manual_final_taxonomy",
        "manual_semantic_equivalent",
        "manual_answer_critical_support_loss",
        "manual_nonannotated_context_useful",
        "manual_paper_location",
        "manual_notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for item in selected:
            c3 = item["variants"]["c3_v3"]
            matched = item["variants"]["mem0_c3_matched"]
            m = item["comparisons"]["c3_minus_mem0_c3_matched"]
            s = item["comparisons"]["c3_minus_simple"]
            t = item["comparisons"]["c3_minus_mem0_top20"]

            writer.writerow(
                {
                    "paper_case_rank": item["paper_case_rank"],
                    "question_id": item["question_id"],
                    "user_id": item["user_id"],
                    "question_type": item["question_type"],
                    "selection_bucket": item["selection_bucket"],
                    "query": item["query"],
                    "gold_answer": item["gold_answer"],
                    "auto_tags": "|".join(item["auto_tags"]),
                    "c3_answer_f1": c3["answer_f1"],
                    "matched_answer_f1": matched["answer_f1"],
                    "delta_answer_f1_c3_minus_matched": m["answer_f1_delta"],
                    "c3_evidence_f1": c3["evidence_f1"],
                    "matched_evidence_f1": matched["evidence_f1"],
                    "delta_evidence_f1_c3_minus_matched": m["evidence_f1_delta"],
                    "c3_evidence_precision": c3["evidence_precision"],
                    "matched_evidence_precision": matched["evidence_precision"],
                    "c3_evidence_recall": c3["evidence_recall"],
                    "matched_evidence_recall": matched["evidence_recall"],
                    "c3_hit": c3["evidence_hit"],
                    "matched_hit": matched["evidence_hit"],
                    "lost_annotated_source_ids": "|".join(
                        m["lost_annotated_source_ids"]
                    ),
                    "gained_annotated_source_ids": "|".join(
                        m["gained_annotated_source_ids"]
                    ),
                    "delta_answer_f1_c3_minus_simple": s["answer_f1_delta"],
                    "delta_answer_f1_c3_minus_mem0_top20": t["answer_f1_delta"],
                    "manual_final_taxonomy": "",
                    "manual_semantic_equivalent": "",
                    "manual_answer_critical_support_loss": "",
                    "manual_nonannotated_context_useful": "",
                    "manual_paper_location": "",
                    "manual_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=1786)
    parser.add_argument("--expected-generations", type=int, default=7144)
    parser.add_argument("--top-cases", type=int, default=30)
    args = parser.parse_args()

    if not 20 <= args.top_cases <= 30:
        raise ValueError("--top-cases must be between 20 and 30")

    predictions = read_jsonl(args.predictions)
    inputs = read_jsonl(args.inputs)

    if len(predictions) != args.expected_generations:
        raise AssertionError(
            f"Expected {args.expected_generations} generations, "
            f"found {len(predictions)}"
        )
    if len(inputs) != args.expected_questions:
        raise AssertionError(
            f"Expected {args.expected_questions} questions, found {len(inputs)}"
        )

    input_map = {str(row["question_id"]): row for row in inputs}
    if len(input_map) != args.expected_questions:
        raise AssertionError("Duplicate question IDs in comparison inputs")

    by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        qid = str(row["question_id"])
        variant = str(row["variant"])
        if variant not in VARIANTS:
            raise AssertionError(f"Unexpected variant: {variant}")
        if variant in by_qid[qid]:
            raise AssertionError(f"Duplicate prediction: {(qid, variant)}")
        by_qid[qid][variant] = row

    if set(by_qid) != set(input_map):
        missing = sorted(set(input_map) - set(by_qid))[:10]
        extra = sorted(set(by_qid) - set(input_map))[:10]
        raise AssertionError(f"Question mismatch missing={missing} extra={extra}")

    for qid, rows in by_qid.items():
        if set(rows) != set(VARIANTS):
            raise AssertionError(
                f"{qid}: variants={sorted(rows)}, expected={list(VARIANTS)}"
            )

    diagnostics: list[dict[str, Any]] = []
    for qid in sorted(input_map):
        diagnostics.append(
            diagnostic_row(qid, input_map[qid], by_qid[qid])
        )

    diagnostics.sort(key=lambda x: str(x["question_id"]))
    selected = select_balanced_cases(diagnostics, args.top_cases)
    summary = build_summary(diagnostics)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.output_dir / "all_question_diagnostics.jsonl"
    selected_jsonl = args.output_dir / "selected_paper_cases.jsonl"
    selected_csv = args.output_dir / "selected_paper_cases_manual_audit.csv"
    selected_md = args.output_dir / "selected_paper_cases.md"
    summary_path = args.output_dir / "error_analysis_summary.json"
    manifest_path = args.output_dir / "ERROR_ANALYSIS_MANIFEST.txt"

    write_jsonl(all_path, diagnostics)
    write_jsonl(selected_jsonl, selected)
    write_selected_csv(selected_csv, selected)
    write_selected_markdown(selected_md, selected)
    write_json(summary_path, summary)

    script_path = Path(__file__).resolve()
    manifest = (
        "LOCOMO ERROR ANALYSIS\n\n"
        "status=POST_HOC_ANALYSIS_ONLY\n"
        "method_modification=none\n"
        "generation_rerun=none\n"
        f"questions={len(diagnostics)}\n"
        f"generations={len(predictions)}\n"
        f"selected_manual_audit_cases={len(selected)}\n"
        "primary_comparison=c3_v3_vs_mem0_c3_matched\n"
        "automatic_taxonomy_status=candidate_labels_only\n\n"
        f"predictions={args.predictions}\n"
        f"predictions_sha256={sha256(args.predictions)}\n"
        f"inputs={args.inputs}\n"
        f"inputs_sha256={sha256(args.inputs)}\n"
        f"analysis_script={script_path}\n"
        f"analysis_script_sha256={sha256(script_path)}\n"
    )
    manifest_path.write_text(manifest, encoding="utf-8")

    checksum_targets = [
        all_path,
        selected_jsonl,
        selected_csv,
        selected_md,
        summary_path,
        manifest_path,
    ]
    checksum_path = args.output_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in checksum_targets
        ),
        encoding="utf-8",
    )

    bucket_counts = Counter(
        item["selection_bucket"] for item in selected
    )

    print("LOCOMO ERROR ANALYSIS: PASS")
    print("questions:", len(diagnostics))
    print("generations:", len(predictions))
    print("selected cases:", len(selected))
    print("selection buckets:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")
    print()
    print("outputs:")
    for path in [
        summary_path,
        selected_md,
        selected_csv,
        selected_jsonl,
        all_path,
        manifest_path,
        checksum_path,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()
