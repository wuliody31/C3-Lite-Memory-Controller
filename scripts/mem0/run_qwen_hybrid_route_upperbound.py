from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import load_config
from src.pipeline import C3Pipeline
from src.schemas import QueryState
from src.transformers_backbone import TransformersBackbone


VARIANTS = (
    "base_hybrid",
    "all_route_hybrid",
)


def load_module(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not path.exists():
        return rows

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8",
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        if not isinstance(value, dict):
            raise TypeError(
                f"{path}:{line_number} is not a JSON object"
            )

        rows.append(value)

    return rows


def append_jsonl(
    handle: Any,
    row: dict[str, Any],
) -> None:
    handle.write(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


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

    separators = (";", "|")

    for separator in separators:
        if separator in text:
            return list(
                dict.fromkeys(
                    item.strip()
                    for item in text.split(separator)
                    if item.strip()
                )
            )

    return [text]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()

    return text in {
        "1",
        "true",
        "yes",
        "y",
        "t",
    }


def answer_references(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item)
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return answer_references(decoded)
        except json.JSONDecodeError:
            pass

    return [text]


def normalize_answer(text: str) -> str:
    lowered = text.lower()

    no_punctuation = "".join(
        " "
        if character in string.punctuation
        else character
        for character in lowered
    )

    no_articles = re.sub(
        r"\b(a|an|the)\b",
        " ",
        no_punctuation,
    )

    return " ".join(
        no_articles.split()
    )


def answer_exact_match(
    prediction: str,
    reference: str,
) -> float:
    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def answer_f1(
    prediction: str,
    reference: str,
) -> float:
    prediction_tokens = normalize_answer(
        prediction
    ).split()

    reference_tokens = normalize_answer(
        reference
    ).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0

    prediction_counter = Counter(
        prediction_tokens
    )
    reference_counter = Counter(
        reference_tokens
    )

    overlap = sum(
        (
            prediction_counter
            & reference_counter
        ).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(
        prediction_tokens
    )
    recall = overlap / len(
        reference_tokens
    )

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def score_answer(
    prediction: str,
    references: list[str],
) -> dict[str, float]:
    if not references:
        return {
            "answer_f1": 0.0,
            "answer_exact_match": 0.0,
        }

    return {
        "answer_f1": max(
            answer_f1(
                prediction,
                reference,
            )
            for reference in references
        ),
        "answer_exact_match": max(
            answer_exact_match(
                prediction,
                reference,
            )
            for reference in references
        ),
    }


def configure_variant(
    base_config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    config = deepcopy(
        base_config
    )

    force_all_route = (
        variant
        == "all_route_hybrid"
    )

    ablation = config.setdefault(
        "ablation",
        {},
    )

    ablation.update(
        {
            "variant": (
                "no_route_planner"
                if force_all_route
                else "full_c3"
            ),
            "disable_route_planner": (
                force_all_route
            ),
            "disable_conflict_handling": False,
            "disable_coverage_confidence_gate": False,
            "disable_evidence_selector": False,
        }
    )

    return config


def build_backbone(
    generation_config: dict[str, Any],
) -> tuple[TransformersBackbone, dict[str, Any]]:
    signature = inspect.signature(
        TransformersBackbone
    )

    model_name = str(
        generation_config[
            "transformers_model"
        ]
    )

    aliases: dict[str, Any] = {
        "model": model_name,
        "model_name": model_name,
        "model_name_or_path": model_name,
        "pretrained_model_name_or_path": model_name,
        "load_in_4bit": bool(
            generation_config.get(
                "transformers_load_in_4bit",
                True,
            )
        ),
        "compute_dtype": str(
            generation_config.get(
                "transformers_compute_dtype",
                "bfloat16",
            )
        ),
        "dtype": str(
            generation_config.get(
                "transformers_compute_dtype",
                "bfloat16",
            )
        ),
        "enable_thinking": bool(
            generation_config.get(
                "transformers_enable_thinking",
                False,
            )
        ),
        "adapter_path": (
            generation_config.get(
                "transformers_adapter_path"
            )
        ),
        "device_map": "auto",
        "trust_remote_code": True,
    }

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for name, parameter in (
        signature.parameters.items()
    ):
        if name == "self":
            continue

        if name in aliases:
            kwargs[name] = aliases[name]
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ):
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            "Unable to construct TransformersBackbone. "
            "Unsupported required parameters: "
            + ", ".join(
                missing_required
            )
            + f". Signature: {signature}"
        )

    backbone = TransformersBackbone(
        **kwargs
    )

    manifest = {
        "class": (
            f"{TransformersBackbone.__module__}."
            f"{TransformersBackbone.__name__}"
        ),
        "signature": str(signature),
        "constructor_kwargs": {
            key: value
            for key, value in kwargs.items()
            if key != "adapter_path"
            or value is not None
        },
        "model": model_name,
        "temperature": float(
            generation_config.get(
                "temperature",
                0.0,
            )
        ),
        "max_new_tokens": int(
            generation_config.get(
                "max_new_tokens",
                256,
            )
        ),
    }

    return backbone, manifest


def bootstrap_mean_delta(
    deltas: list[float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if not deltas:
        return {
            "mean_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_two_sided": 1.0,
        }

    observed = sum(
        deltas
    ) / len(
        deltas
    )

    rng = random.Random(
        seed
    )
    samples: list[float] = []
    n = len(
        deltas
    )

    for _ in range(
        repetitions
    ):
        sample_mean = sum(
            deltas[
                rng.randrange(n)
            ]
            for _ in range(n)
        ) / n

        samples.append(
            sample_mean
        )

    samples.sort()

    low_index = max(
        0,
        int(
            math.floor(
                0.025
                * repetitions
            )
        ),
    )
    high_index = min(
        repetitions - 1,
        int(
            math.ceil(
                0.975
                * repetitions
            )
        )
        - 1,
    )

    non_positive = sum(
        sample <= 0.0
        for sample in samples
    )
    non_negative = sum(
        sample >= 0.0
        for sample in samples
    )

    p_value = min(
        1.0,
        2.0
        * min(
            non_positive,
            non_negative,
        )
        / repetitions,
    )

    return {
        "mean_delta": observed,
        "ci_low": samples[
            low_index
        ],
        "ci_high": samples[
            high_index
        ],
        "p_two_sided": p_value,
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["variant"] == variant
    ]

    if not selected:
        raise AssertionError(
            f"No rows for {variant}"
        )

    answerable = [
        row
        for row in selected
        if not row[
            "should_abstain"
        ]
    ]

    unanswerable = [
        row
        for row in selected
        if row[
            "should_abstain"
        ]
    ]

    def average(
        field: str,
        subset: list[
            dict[str, Any]
        ],
    ) -> float:
        if not subset:
            return 0.0

        return sum(
            float(
                row[field]
            )
            for row in subset
        ) / len(
            subset
        )

    return {
        "questions": len(
            selected
        ),
        "answerable_questions": len(
            answerable
        ),
        "unanswerable_questions": len(
            unanswerable
        ),
        "answer_f1_all": average(
            "answer_f1",
            selected,
        ),
        "answer_f1_answerable": average(
            "answer_f1",
            answerable,
        ),
        "answer_exact_match_all": average(
            "answer_exact_match",
            selected,
        ),
        "evidence_precision": average(
            "evidence_precision",
            selected,
        ),
        "evidence_recall": average(
            "evidence_recall",
            selected,
        ),
        "evidence_f1": average(
            "evidence_f1",
            selected,
        ),
        "evidence_hit_rate": average(
            "evidence_hit",
            selected,
        ),
        "mean_selected": average(
            "selected_count",
            selected,
        ),
        "mean_latency_ms": average(
            "latency_ms",
            selected,
        ),
        "mean_input_tokens": average(
            "input_tokens_numeric",
            selected,
        ),
        "mean_output_tokens": average(
            "output_tokens_numeric",
            selected,
        ),
        "abstention_accuracy": average(
            "abstention_correct",
            selected,
        ),
        "unanswerable_abstain_recall": average(
            "predicted_abstain",
            unanswerable,
        ),
        "answerable_response_rate": average(
            "predicted_response",
            answerable,
        ),
    }


def category_breakdown(
    rows: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        if row[
            "variant"
        ] != variant:
            continue

        grouped[
            str(
                row[
                    "question_type"
                ]
            )
        ].append(
            row
        )

    output: dict[str, Any] = {}

    for category, items in sorted(
        grouped.items()
    ):
        output[
            category
        ] = {
            "questions": len(
                items
            ),
            "answer_f1": sum(
                float(
                    item[
                        "answer_f1"
                    ]
                )
                for item in items
            )
            / len(
                items
            ),
            "evidence_f1": sum(
                float(
                    item[
                        "evidence_f1"
                    ]
                )
                for item in items
            )
            / len(
                items
            ),
            "evidence_hit_rate": sum(
                float(
                    item[
                        "evidence_hit"
                    ]
                )
                for item in items
            )
            / len(
                items
            ),
        }

    return output


def build_report(
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Qwen3 Hybrid Route Upper-Bound — LoCoMo Pilot-200",
        "",
        "## Protocol",
        "",
        "- Backbone: Qwen/Qwen3-8B, 4-bit NF4, frozen.",
        "- Temperature: 0.0.",
        "- Candidate pool: frozen C3 raw candidates union Mem0 top-20.",
        "- `base_hybrid`: current C3 route.",
        "- `all_route_hybrid`: all memory types enabled.",
        "- Ranker, candidate budget, conflict handling, evidence selector, "
        "confidence controller and generation settings otherwise unchanged.",
        "",
        "## Main results",
        "",
        "| Variant | Answer F1 | Answerable F1 | Evidence P | Evidence R | "
        "Evidence F1 | Hit | Mean selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = summary[
            "variants"
        ][
            variant
        ]

        lines.append(
            f"| {variant} | "
            f"{row['answer_f1_all']:.4f} | "
            f"{row['answer_f1_answerable']:.4f} | "
            f"{row['evidence_precision']:.4f} | "
            f"{row['evidence_recall']:.4f} | "
            f"{row['evidence_f1']:.4f} | "
            f"{row['evidence_hit_rate']:.4f} | "
            f"{row['mean_selected']:.2f} |"
        )

    paired = summary[
        "paired_statistics"
    ]

    lines.extend(
        [
            "",
            "## Paired bootstrap: all-route minus base",
            "",
            "| Metric | Delta | 95% CI | p-value |",
            "|---|---:|---:|---:|",
            (
                "| Answer F1 | "
                f"{paired['answer_f1']['mean_delta']:+.4f} | "
                f"[{paired['answer_f1']['ci_low']:+.4f}, "
                f"{paired['answer_f1']['ci_high']:+.4f}] | "
                f"{paired['answer_f1']['p_two_sided']:.6f} |"
            ),
            (
                "| Evidence F1 | "
                f"{paired['evidence_f1']['mean_delta']:+.4f} | "
                f"[{paired['evidence_f1']['ci_low']:+.4f}, "
                f"{paired['evidence_f1']['ci_high']:+.4f}] | "
                f"{paired['evidence_f1']['p_two_sided']:.6f} |"
            ),
            "",
            "## Interpretation",
            "",
            (
                "This is an upper-bound routing experiment. A positive "
                "all-route result supports implementing query-adaptive "
                "route expansion; it does not justify permanently retrieving "
                "all memory types in the final controller."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-8B generation for base-hybrid and all-route-hybrid "
            "on the fixed LoCoMo Pilot-200 union candidate pool."
        )
    )

    parser.add_argument(
        "--runner-module",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--memories",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--c3-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mem0-top20",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260803,
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help=(
            "Zero means all questions. Non-zero is useful only for smoke tests."
        ),
    )
    parser.add_argument(
        "--skip-backbone-smoke",
        action="store_true",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runner = load_module(
        args.runner_module,
        "hybrid_runner_module",
    )

    base_config = load_config(
        args.config
    )

    generation_config = (
        base_config[
            "generation"
        ]
    )

    records = runner.load_memory_records(
        args.memories
    )
    catalog, source_map = (
        runner.build_catalog(
            records
        )
    )

    questions = read_csv(
        args.questions
    )
    questions_by_id = {
        str(
            row[
                "question_id"
            ]
        ): row
        for row in questions
    }

    c3_rows = [
        row
        for row in read_jsonl(
            args.c3_predictions
        )
        if str(
            row.get(
                "method",
                "",
            )
        ).lower()
        in {
            "c3",
            "c3_lite_controller",
        }
    ]
    c3_by_id = {
        str(
            row[
                "question_id"
            ]
        ): row
        for row in c3_rows
    }

    mem0_rows = read_jsonl(
        args.mem0_top20
    )
    mem0_by_id = {
        str(
            row[
                "question_id"
            ]
        ): row
        for row in mem0_rows
    }

    question_ids = sorted(
        set(
            questions_by_id
        )
    )

    if len(
        question_ids
    ) != 200:
        raise AssertionError(
            "Expected 200 canonical Pilot questions."
        )

    if set(
        c3_by_id
    ) != set(
        question_ids
    ):
        raise AssertionError(
            "C3 IDs do not match canonical questions."
        )

    if set(
        mem0_by_id
    ) != set(
        question_ids
    ):
        raise AssertionError(
            "Mem0 IDs do not match canonical questions."
        )

    if args.max_questions > 0:
        question_ids = question_ids[
            : args.max_questions
        ]

    hybrid_pools = {}

    missing_catalog: set[str] = set()

    for question_id in question_ids:
        c3_ids = split_ids(
            c3_by_id[
                question_id
            ].get(
                "raw_retrieved_ids"
            )
        )

        mem0_ids = split_ids(
            mem0_by_id[
                question_id
            ].get(
                "retrieved_memory_ids"
            )
        )

        for memory_id in [
            *c3_ids,
            *mem0_ids,
        ]:
            if memory_id not in catalog:
                missing_catalog.add(
                    memory_id
                )

        hybrid_pools[
            question_id
        ] = runner.make_pool(
            c3_ids,
            mem0_ids,
        )

    if missing_catalog:
        raise AssertionError(
            f"{len(missing_catalog)} candidate IDs "
            "are absent from the canonical memory catalog. "
            f"Examples: {sorted(missing_catalog)[:20]}"
        )

    print(
        "Loading Qwen3 backbone...",
        flush=True,
    )

    backbone_start = time.perf_counter()

    backbone, backbone_manifest = (
        build_backbone(
            generation_config
        )
    )

    backbone_manifest[
        "construction_seconds"
    ] = (
        time.perf_counter()
        - backbone_start
    )

    if not args.skip_backbone_smoke:
        smoke_start = time.perf_counter()

        smoke_result = backbone.generate(
            "Reply with exactly: BACKBONE_READY",
            temperature=0.0,
            max_new_tokens=16,
        )

        backbone_manifest[
            "smoke"
        ] = {
            "text": smoke_result.text,
            "input_tokens": (
                smoke_result.input_tokens
            ),
            "output_tokens": (
                smoke_result.output_tokens
            ),
            "latency_seconds": (
                time.perf_counter()
                - smoke_start
            ),
        }

        print(
            "Backbone smoke response:",
            smoke_result.text,
            flush=True,
        )

    (
        args.output_dir
        / "backbone_manifest.json"
    ).write_text(
        json.dumps(
            backbone_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pipelines = {}

    for variant in VARIANTS:
        pipelines[
            variant
        ] = C3Pipeline(
            config=configure_variant(
                base_config,
                variant,
            ),
            memory_store=(
                runner.FrozenCandidateStore(
                    catalog=catalog,
                    pools=hybrid_pools,
                    label=variant,
                )
            ),
            procedure_store=None,
            backbone=backbone,
            prompt_template=(
                args.prompt_template
            ),
        )

    predictions_path = (
        args.output_dir
        / "predictions.jsonl"
    )

    existing_rows = read_jsonl(
        predictions_path
    )

    completed = {
        (
            str(
                row[
                    "question_id"
                ]
            ),
            str(
                row[
                    "variant"
                ]
            ),
        )
        for row in existing_rows
    }

    expected_keys = {
        (
            question_id,
            variant,
        )
        for question_id in question_ids
        for variant in VARIANTS
    }

    unknown_keys = (
        completed
        - expected_keys
    )

    if unknown_keys:
        raise AssertionError(
            "Predictions file contains unexpected rows: "
            f"{sorted(unknown_keys)[:10]}"
        )

    print(
        f"Resume state: {len(completed)}/"
        f"{len(expected_keys)} generations complete.",
        flush=True,
    )

    try:
        with predictions_path.open(
            "a",
            encoding="utf-8",
        ) as output_handle:
            total = len(
                expected_keys
            )

            for question_index, question_id in enumerate(
                question_ids,
                start=1,
            ):
                c3 = c3_by_id[
                    question_id
                ]
                canonical = questions_by_id[
                    question_id
                ]

                query = str(
                    c3[
                        "query"
                    ]
                )
                user_id = str(
                    c3[
                        "user_id"
                    ]
                )
                question_type = str(
                    c3.get(
                        "question_type",
                        canonical.get(
                            "question_type",
                            "unknown",
                        ),
                    )
                )

                gold_memory_ids = split_ids(
                    canonical.get(
                        "supporting_memory_ids",
                        "",
                    )
                )
                gold_source_ids = (
                    runner.project_ids(
                        gold_memory_ids,
                        source_map,
                    )
                )

                references = answer_references(
                    canonical.get(
                        "gold_answer",
                        c3.get(
                            "gold_answer"
                        ),
                    )
                )

                if not references:
                    references = answer_references(
                        c3.get(
                            "gold_answer"
                        )
                    )

                should_abstain = parse_bool(
                    canonical.get(
                        "should_abstain",
                        c3.get(
                            "should_abstain",
                            False,
                        ),
                    )
                )

                shared_time = datetime.now(
                    timezone.utc
                )

                for variant in VARIANTS:
                    key = (
                        question_id,
                        variant,
                    )

                    if key in completed:
                        continue

                    print(
                        f"[{len(completed) + 1}/{total}] "
                        f"question={question_index}/{len(question_ids)} "
                        f"id={question_id} variant={variant}",
                        flush=True,
                    )

                    result = pipelines[
                        variant
                    ].answer(
                        QueryState(
                            query=query,
                            user_id=user_id,
                            session_id=(
                                question_id
                            ),
                            current_time=(
                                shared_time
                            ),
                        )
                    )

                    selected_ids = list(
                        result.selected_ids
                    )
                    selected_source_ids = (
                        runner.project_ids(
                            selected_ids,
                            source_map,
                        )
                    )

                    evidence = (
                        runner.evidence_metrics(
                            selected_source_ids,
                            gold_source_ids,
                        )
                    )

                    answer_scores = score_answer(
                        result.answer,
                        references,
                    )

                    predicted_abstain = (
                        result.decision.value
                        == "abstain"
                    )

                    row = {
                        "question_id": (
                            question_id
                        ),
                        "variant": variant,
                        "question_type": (
                            question_type
                        ),
                        "query": query,
                        "user_id": user_id,
                        "gold_answer": (
                            references
                        ),
                        "should_abstain": (
                            should_abstain
                        ),
                        "answer": (
                            result.answer
                        ),
                        "decision": (
                            result.decision.value
                        ),
                        "predicted_abstain": (
                            predicted_abstain
                        ),
                        "predicted_response": (
                            not predicted_abstain
                        ),
                        "abstention_correct": (
                            predicted_abstain
                            == should_abstain
                        ),
                        **answer_scores,
                        "gold_memory_ids": (
                            gold_memory_ids
                        ),
                        "gold_source_ids": (
                            gold_source_ids
                        ),
                        "raw_retrieved_ids": (
                            list(
                                result.raw_retrieved_ids
                            )
                        ),
                        "ranked_candidate_ids": (
                            list(
                                result.ranked_candidate_ids
                            )
                        ),
                        "selected_ids": (
                            selected_ids
                        ),
                        "selected_source_ids": (
                            selected_source_ids
                        ),
                        "selected_evidence": [
                            candidate.to_dict()
                            for candidate in (
                                result.selected_evidence
                            )
                        ],
                        "selected_memory_types": (
                            result.selected_memory_types
                        ),
                        "route_scores": (
                            result.route_scores
                        ),
                        "evidence_precision": (
                            evidence[
                                "precision"
                            ]
                        ),
                        "evidence_recall": (
                            evidence[
                                "recall"
                            ]
                        ),
                        "evidence_f1": (
                            evidence[
                                "f1"
                            ]
                        ),
                        "evidence_hit": (
                            evidence[
                                "hit"
                            ]
                        ),
                        "evidence_overlap_source_ids": (
                            evidence[
                                "overlap_source_ids"
                            ]
                        ),
                        "selected_count": len(
                            selected_ids
                        ),
                        "latency_ms": (
                            result.latency_ms
                        ),
                        "input_tokens": (
                            result.input_tokens
                        ),
                        "output_tokens": (
                            result.output_tokens
                        ),
                        "input_tokens_numeric": (
                            result.input_tokens
                            or 0
                        ),
                        "output_tokens_numeric": (
                            result.output_tokens
                            or 0
                        ),
                    }

                    append_jsonl(
                        output_handle,
                        row,
                    )

                    completed.add(
                        key
                    )
    finally:
        for pipeline in (
            pipelines.values()
        ):
            pipeline.close()

    final_rows = read_jsonl(
        predictions_path
    )

    by_key = {}

    for row in final_rows:
        key = (
            str(
                row[
                    "question_id"
                ]
            ),
            str(
                row[
                    "variant"
                ]
            ),
        )
        by_key[
            key
        ] = row

    if set(
        by_key
    ) != expected_keys:
        missing = sorted(
            expected_keys
            - set(
                by_key
            )
        )

        raise AssertionError(
            f"Incomplete generation. Missing: {missing[:20]}"
        )

    final_rows = [
        by_key[
            (
                question_id,
                variant,
            )
        ]
        for question_id in question_ids
        for variant in VARIANTS
    ]

    variant_summaries = {
        variant: aggregate_rows(
            final_rows,
            variant,
        )
        for variant in VARIANTS
    }

    base_by_id = {
        row[
            "question_id"
        ]: row
        for row in final_rows
        if row[
            "variant"
        ]
        == "base_hybrid"
    }
    all_route_by_id = {
        row[
            "question_id"
        ]: row
        for row in final_rows
        if row[
            "variant"
        ]
        == "all_route_hybrid"
    }

    answer_deltas = [
        float(
            all_route_by_id[
                question_id
            ][
                "answer_f1"
            ]
        )
        - float(
            base_by_id[
                question_id
            ][
                "answer_f1"
            ]
        )
        for question_id in question_ids
    ]

    evidence_deltas = [
        float(
            all_route_by_id[
                question_id
            ][
                "evidence_f1"
            ]
        )
        - float(
            base_by_id[
                question_id
            ][
                "evidence_f1"
            ]
        )
        for question_id in question_ids
    ]

    answer_better = sum(
        delta > 0.0
        for delta in answer_deltas
    )
    answer_worse = sum(
        delta < 0.0
        for delta in answer_deltas
    )
    answer_tied = len(
        answer_deltas
    ) - answer_better - answer_worse

    summary = {
        "experiment": (
            "rc8_8b8_qwen3_route_upperbound"
        ),
        "questions": len(
            question_ids
        ),
        "generations": len(
            final_rows
        ),
        "backbone": (
            backbone_manifest
        ),
        "variants": (
            variant_summaries
        ),
        "paired_outcomes": {
            "answer_f1_better": (
                answer_better
            ),
            "answer_f1_worse": (
                answer_worse
            ),
            "answer_f1_tied": (
                answer_tied
            ),
        },
        "paired_statistics": {
            "answer_f1": (
                bootstrap_mean_delta(
                    answer_deltas,
                    repetitions=(
                        args.bootstrap_repetitions
                    ),
                    seed=(
                        args.bootstrap_seed
                    ),
                )
            ),
            "evidence_f1": (
                bootstrap_mean_delta(
                    evidence_deltas,
                    repetitions=(
                        args.bootstrap_repetitions
                    ),
                    seed=(
                        args.bootstrap_seed
                        + 1
                    ),
                )
            ),
        },
        "category_breakdown": {
            variant: category_breakdown(
                final_rows,
                variant,
            )
            for variant in VARIANTS
        },
    }

    (
        args.output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir
        / "report.md"
    ).write_text(
        build_report(
            summary
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        "QWEN3 HYBRID ROUTE UPPER-BOUND: PASSED"
    )


if __name__ == "__main__":
    main()
