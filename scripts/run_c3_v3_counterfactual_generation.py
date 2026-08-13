from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import load_config
from src.transformers_backbone import TransformersBackbone


VARIANTS = ("legacy", "c3_v3")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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
        "adapter_path": generation_config.get(
            "transformers_adapter_path"
        ),
        "device_map": "auto",
        "trust_remote_code": True,
    }

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []

    for name, parameter in signature.parameters.items():
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
            + ", ".join(missing_required)
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
    }

    return backbone, manifest


def average(
    rows: list[dict[str, Any]],
    field: str,
) -> float:
    if not rows:
        return 0.0

    return sum(
        float(row.get(field, 0.0) or 0.0)
        for row in rows
    ) / len(rows)


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

    observed = sum(deltas) / len(deltas)

    rng = random.Random(seed)
    n = len(deltas)
    samples: list[float] = []

    for _ in range(repetitions):
        sample_mean = sum(
            deltas[
                rng.randrange(n)
            ]
            for _ in range(n)
        ) / n
        samples.append(sample_mean)

    samples.sort()

    low_index = max(
        0,
        int(math.floor(0.025 * repetitions)),
    )

    high_index = min(
        repetitions - 1,
        int(math.ceil(0.975 * repetitions)) - 1,
    )

    non_positive = sum(
        value <= 0.0
        for value in samples
    )

    non_negative = sum(
        value >= 0.0
        for value in samples
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
        "ci_low": samples[low_index],
        "ci_high": samples[high_index],
        "p_two_sided": p_value,
    }


def aggregate_variant(
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
            f"No rows for variant={variant}"
        )

    changed = [
        row
        for row in selected
        if row["evidence_set_changed"]
    ]

    removed_gold = [
        row
        for row in selected
        if row["removed_gold_count"] > 0
    ]

    return {
        "questions": len(selected),
        "answer_f1": average(
            selected,
            "answer_f1",
        ),
        "answer_exact_match": average(
            selected,
            "answer_exact_match",
        ),
        "mean_input_tokens": average(
            selected,
            "input_tokens_numeric",
        ),
        "mean_output_tokens": average(
            selected,
            "output_tokens_numeric",
        ),
        "mean_latency_ms": average(
            selected,
            "latency_ms",
        ),
        "changed_evidence_subset": {
            "questions": len(changed),
            "answer_f1": average(
                changed,
                "answer_f1",
            ),
            "answer_exact_match": average(
                changed,
                "answer_exact_match",
            ),
        },
        "removed_gold_subset": {
            "questions": len(removed_gold),
            "answer_f1": average(
                removed_gold,
                "answer_f1",
            ),
            "answer_exact_match": average(
                removed_gold,
                "answer_exact_match",
            ),
        },
    }


def paired_rows(
    rows: list[dict[str, Any]],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any],
    ]
]:
    by_key = {
        (
            str(row["question_id"]),
            str(row["variant"]),
        ): row
        for row in rows
    }

    question_ids = sorted(
        {
            str(row["question_id"])
            for row in rows
        }
    )

    output = []

    for question_id in question_ids:
        legacy_key = (
            question_id,
            "legacy",
        )
        c3_key = (
            question_id,
            "c3_v3",
        )

        if (
            legacy_key not in by_key
            or c3_key not in by_key
        ):
            continue

        output.append(
            (
                by_key[legacy_key],
                by_key[c3_key],
            )
        )

    return output


def build_summary(
    rows: list[dict[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs = paired_rows(rows)

    if not pairs:
        raise AssertionError(
            "No complete Legacy/C3-v3 pairs found."
        )

    all_answer_f1_deltas = [
        float(c3["answer_f1"])
        - float(legacy["answer_f1"])
        for legacy, c3 in pairs
    ]

    changed_pairs = [
        (legacy, c3)
        for legacy, c3 in pairs
        if bool(c3["evidence_set_changed"])
    ]

    changed_answer_f1_deltas = [
        float(c3["answer_f1"])
        - float(legacy["answer_f1"])
        for legacy, c3 in changed_pairs
    ]

    removed_gold_pairs = [
        (legacy, c3)
        for legacy, c3 in pairs
        if int(c3["removed_gold_count"]) > 0
    ]

    removed_gold_answer_f1_deltas = [
        float(c3["answer_f1"])
        - float(legacy["answer_f1"])
        for legacy, c3 in removed_gold_pairs
    ]

    better = sum(
        delta > 0.0
        for delta in all_answer_f1_deltas
    )
    worse = sum(
        delta < 0.0
        for delta in all_answer_f1_deltas
    )

    changed_better = sum(
        delta > 0.0
        for delta in changed_answer_f1_deltas
    )
    changed_worse = sum(
        delta < 0.0
        for delta in changed_answer_f1_deltas
    )

    removed_gold_better = sum(
        delta > 0.0
        for delta in removed_gold_answer_f1_deltas
    )
    removed_gold_worse = sum(
        delta < 0.0
        for delta in removed_gold_answer_f1_deltas
    )

    return {
        "questions": len(pairs),
        "generations": len(rows),
        "variants": {
            variant: aggregate_variant(
                rows,
                variant,
            )
            for variant in VARIANTS
        },
        "paired_answer_f1": {
            "all_questions": {
                **bootstrap_mean_delta(
                    all_answer_f1_deltas,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                ),
                "c3_better": better,
                "c3_worse": worse,
                "tied": (
                    len(all_answer_f1_deltas)
                    - better
                    - worse
                ),
            },
            "changed_evidence_questions": {
                "questions": len(
                    changed_answer_f1_deltas
                ),
                **bootstrap_mean_delta(
                    changed_answer_f1_deltas,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 1,
                ),
                "c3_better": changed_better,
                "c3_worse": changed_worse,
                "tied": (
                    len(changed_answer_f1_deltas)
                    - changed_better
                    - changed_worse
                ),
            },
            "removed_gold_questions": {
                "questions": len(
                    removed_gold_answer_f1_deltas
                ),
                **bootstrap_mean_delta(
                    removed_gold_answer_f1_deltas,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 2,
                ),
                "c3_better": removed_gold_better,
                "c3_worse": removed_gold_worse,
                "tied": (
                    len(removed_gold_answer_f1_deltas)
                    - removed_gold_better
                    - removed_gold_worse
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Qwen3 paired counterfactual generation for "
            "Legacy evidence versus C3-v3 shadow evidence."
        )
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help=(
            "0 means all questions. Use 2-3 only for a GPU smoke test."
        ),
    )

    parser.add_argument(
        "--skip-backbone-smoke",
        action="store_true",
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260812,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_rows = read_jsonl(
        args.pairs
    )

    if len(pair_rows) != 60:
        raise AssertionError(
            f"Expected 60 paired Dataset A questions, got {len(pair_rows)}."
        )

    if args.max_questions > 0:
        pair_rows = pair_rows[
            : args.max_questions
        ]

    config = load_config(
        args.config
    )

    generation_config = config[
        "generation"
    ]

    temperature = float(
        generation_config.get(
            "temperature",
            0.0,
        )
    )

    max_new_tokens = int(
        generation_config.get(
            "max_new_tokens",
            256,
        )
    )

    if temperature != 0.0:
        raise AssertionError(
            "Counterfactual audit requires frozen temperature=0.0."
        )

    print(
        "Loading Qwen3 backbone...",
        flush=True,
    )

    load_start = time.perf_counter()

    backbone, backbone_manifest = build_backbone(
        generation_config
    )

    backbone_manifest.update(
        {
            "construction_seconds": (
                time.perf_counter()
                - load_start
            ),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "thinking_disabled": not bool(
                generation_config.get(
                    "transformers_enable_thinking",
                    False,
                )
            ),
            "experiment": (
                "c3_v3_counterfactual_answer_audit"
            ),
            "paired_variants": list(
                VARIANTS
            ),
        }
    )

    if not args.skip_backbone_smoke:
        smoke_start = time.perf_counter()

        smoke = backbone.generate(
            "Reply with exactly: BACKBONE_READY",
            temperature=0.0,
            max_new_tokens=16,
        )

        backbone_manifest["smoke"] = {
            "text": smoke.text,
            "input_tokens": smoke.input_tokens,
            "output_tokens": smoke.output_tokens,
            "latency_seconds": (
                time.perf_counter()
                - smoke_start
            ),
        }

        print(
            "Backbone smoke response:",
            smoke.text,
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

    predictions_path = (
        args.output_dir
        / "predictions.jsonl"
    )

    existing = read_jsonl(
        predictions_path
    )

    completed = {
        (
            str(row["question_id"]),
            str(row["variant"]),
        )
        for row in existing
    }

    target_generations = (
        len(pair_rows)
        * len(VARIANTS)
    )

    print(
        f"Resume state: {len(completed)}/{target_generations} generations complete.",
        flush=True,
    )

    with predictions_path.open(
        "a",
        encoding="utf-8",
    ) as output_handle:
        generation_index = len(completed)

        for question_index, pair in enumerate(
            pair_rows,
            start=1,
        ):
            question_id = str(
                pair["question_id"]
            )

            legacy_ids = [
                str(item)
                for item in (
                    pair.get(
                        "legacy_selected_ids"
                    )
                    or []
                )
            ]

            c3_ids = [
                str(item)
                for item in (
                    pair.get(
                        "c3_v3_selected_ids"
                    )
                    or []
                )
            ]

            removed_ids = [
                str(item)
                for item in (
                    pair.get(
                        "removed_by_c3_v3"
                    )
                    or []
                )
            ]

            gold_ids = {
                str(item)
                for item in (
                    pair.get(
                        "gold_memory_ids"
                    )
                    or []
                )
            }

            removed_gold_ids = [
                memory_id
                for memory_id in removed_ids
                if memory_id in gold_ids
            ]

            evidence_set_changed = (
                set(legacy_ids)
                != set(c3_ids)
            )

            references = answer_references(
                pair.get(
                    "gold_answer"
                )
            )

            for variant in VARIANTS:
                key = (
                    question_id,
                    variant,
                )

                if key in completed:
                    continue

                prompt_field = (
                    "legacy_prompt"
                    if variant == "legacy"
                    else "c3_v3_prompt"
                )

                selected_ids = (
                    legacy_ids
                    if variant == "legacy"
                    else c3_ids
                )

                prompt = str(
                    pair[
                        prompt_field
                    ]
                )

                generation_index += 1

                print(
                    f"[{generation_index}/{target_generations}] "
                    f"question={question_index}/{len(pair_rows)} "
                    f"id={question_id} variant={variant}",
                    flush=True,
                )

                generation_start = (
                    time.perf_counter()
                )

                result = backbone.generate(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )

                latency_ms = (
                    time.perf_counter()
                    - generation_start
                ) * 1000.0

                scores = score_answer(
                    result.text,
                    references,
                )

                output_row = {
                    "question_id": question_id,
                    "question_type": pair.get(
                        "question_type"
                    ),
                    "query_mode": pair.get(
                        "query_mode"
                    ),
                    "query": pair.get(
                        "query",
                        "",
                    ),
                    "variant": variant,
                    "answer": result.text,
                    "gold_answer": pair.get(
                        "gold_answer",
                        "",
                    ),
                    "should_abstain": bool(
                        pair.get(
                            "should_abstain",
                            False,
                        )
                    ),
                    "selected_ids": selected_ids,
                    "legacy_selected_ids": (
                        legacy_ids
                    ),
                    "c3_v3_selected_ids": c3_ids,
                    "removed_by_c3_v3": (
                        removed_ids
                    ),
                    "removed_gold_ids": (
                        removed_gold_ids
                    ),
                    "removed_gold_count": len(
                        removed_gold_ids
                    ),
                    "evidence_set_changed": (
                        evidence_set_changed
                    ),
                    "hard_complete": bool(
                        pair.get(
                            "hard_complete",
                            False,
                        )
                    ),
                    "answer_f1": scores[
                        "answer_f1"
                    ],
                    "answer_exact_match": scores[
                        "answer_exact_match"
                    ],
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "input_tokens_numeric": (
                        int(result.input_tokens)
                        if result.input_tokens
                        is not None
                        else 0
                    ),
                    "output_tokens_numeric": (
                        int(result.output_tokens)
                        if result.output_tokens
                        is not None
                        else 0
                    ),
                    "latency_ms": latency_ms,
                }

                append_jsonl(
                    output_handle,
                    output_row,
                )

                completed.add(key)

    final_rows = read_jsonl(
        predictions_path
    )

    expected_keys = {
        (
            str(pair["question_id"]),
            variant,
        )
        for pair in pair_rows
        for variant in VARIANTS
    }

    final_rows = [
        row
        for row in final_rows
        if (
            str(row.get("question_id", "")),
            str(row.get("variant", "")),
        )
        in expected_keys
    ]

    final_keys = {
        (
            str(row["question_id"]),
            str(row["variant"]),
        )
        for row in final_rows
    }

    if final_keys != expected_keys:
        missing = sorted(
            expected_keys
            - final_keys
        )

        raise AssertionError(
            "Generation output is incomplete. "
            f"Missing examples: {missing[:10]}"
        )

    summary = build_summary(
        final_rows,
        bootstrap_repetitions=(
            args.bootstrap_repetitions
        ),
        bootstrap_seed=(
            args.bootstrap_seed
        ),
    )

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

    print()
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
