#!/usr/bin/env python3
"""Build Controlled Lifecycle Benchmark v0.1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "lifecycle_benchmark_v01"

CASES_PATH = OUT / "cases.jsonl"
MANIFEST_PATH = OUT / "manifest.json"
README_PATH = OUT / "README.md"


STATE_SPECS = {
    "user.residence": {
        "label": "residence",
        "values": [
            "Nottingham",
            "London",
            "Edinburgh",
        ],
        "event_templates": [
            "The user lives in {value}.",
            "The user's current residence is {value}.",
            "The user is now based in {value}.",
            "The user currently resides in {value}.",
        ],
        "current_queries": [
            "Where does the user live now?",
            "What is the user's current residence?",
            "Where is the user currently based?",
            "Which city does the user currently live in?",
        ],
        "history_queries": [
            "Where did the user live previously?",
            "What was the user's previous residence?",
            "Where was the user based before the current residence?",
            "Which city did the user live in before now?",
        ],
    },
    "user.employer": {
        "label": "employer",
        "values": [
            "Northstar Labs",
            "Meridian Systems",
            "Atlas Robotics",
        ],
        "event_templates": [
            "The user works at {value}.",
            "The user's employer is {value}.",
            "The user is currently employed by {value}.",
            "The user currently works for {value}.",
        ],
        "current_queries": [
            "Where does the user work now?",
            "Who is the user's current employer?",
            "Which company currently employs the user?",
            "What company does the user currently work for?",
        ],
        "history_queries": [
            "Where did the user work previously?",
            "Who was the user's previous employer?",
            "Which company employed the user before the current one?",
            "What company did the user work for before now?",
        ],
    },
    "user.preferred_language": {
        "label": "preferred programming language",
        "values": [
            "Python",
            "Java",
            "Rust",
        ],
        "event_templates": [
            "The user's preferred programming language is {value}.",
            "The user currently prefers {value} for programming.",
            "The user's current coding-language preference is {value}.",
            "For programming, the user now prefers {value}.",
        ],
        "current_queries": [
            "What programming language does the user prefer now?",
            "What is the user's current preferred programming language?",
            "Which programming language does the user currently favour?",
            "Which language does the user currently prefer for coding?",
        ],
        "history_queries": [
            "What programming language did the user prefer previously?",
            "What was the user's previous programming-language preference?",
            "Which programming language did the user favour before the current one?",
            "Which language did the user prefer for coding before now?",
        ],
    },
    "user.subscription_plan": {
        "label": "subscription plan",
        "values": [
            "Basic",
            "Plus",
            "Pro",
        ],
        "event_templates": [
            "The user's subscription plan is {value}.",
            "The user is currently on the {value} plan.",
            "The user's current subscription tier is {value}.",
            "The user currently has a {value} subscription.",
        ],
        "current_queries": [
            "What subscription plan does the user have now?",
            "What is the user's current subscription tier?",
            "Which plan is the user currently on?",
            "What subscription does the user currently have?",
        ],
        "history_queries": [
            "What subscription plan did the user have previously?",
            "What was the user's previous subscription tier?",
            "Which plan was the user on before the current one?",
            "What subscription did the user have before now?",
        ],
    },
    "user.current_project_tool": {
        "label": "project tool",
        "values": [
            "NetworkX",
            "Neo4j",
            "PostgreSQL",
        ],
        "event_templates": [
            "The user's current project tool is {value}.",
            "The user is currently using {value} for the project.",
            "The project's current tool is {value}.",
            "The user now uses {value} in the project.",
        ],
        "current_queries": [
            "What tool is the user using for the project now?",
            "What is the user's current project tool?",
            "Which tool does the project currently use?",
            "What does the user currently use in the project?",
        ],
        "history_queries": [
            "What project tool did the user use previously?",
            "What was the user's previous project tool?",
            "Which tool did the project use before the current one?",
            "What did the user use in the project before now?",
        ],
    },
}


PATTERNS = (
    "replacement",
    "duplicate",
    "multi_step",
    "out_of_order",
    "reversion",
)


def event(
    *,
    event_id: str,
    ingestion_index: int,
    observed_at: str,
    value: str,
    text: str,
) -> dict:
    return {
        "event_id": event_id,
        "ingestion_index": ingestion_index,
        "observed_at": observed_at,
        "value": value,
        "text": text,
    }


def build_pattern(
    *,
    pattern: str,
    values: list[str],
    template: str,
) -> dict:
    a, b, c = values

    def text(value: str) -> str:
        return template.format(value=value)

    if pattern == "replacement":
        events = [
            event(
                event_id="e1",
                ingestion_index=1,
                observed_at="2026-01-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
            event(
                event_id="e2",
                ingestion_index=2,
                observed_at="2026-02-01T09:00:00+00:00",
                value=b,
                text=text(b),
            ),
        ]

        operations = [
            "add_initial",
            "supersede_state",
        ]

        current_gold = b
        historical_gold = [a]
        history_values_expected = [a]
        stale_values = [a]

    elif pattern == "duplicate":
        events = [
            event(
                event_id="e1",
                ingestion_index=1,
                observed_at="2026-01-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
            event(
                event_id="e2",
                ingestion_index=2,
                observed_at="2026-01-02T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
        ]

        operations = [
            "add_initial",
            "noop_duplicate",
        ]

        current_gold = a
        historical_gold = []
        history_values_expected = []
        stale_values = []

    elif pattern == "multi_step":
        events = [
            event(
                event_id="e1",
                ingestion_index=1,
                observed_at="2026-01-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
            event(
                event_id="e2",
                ingestion_index=2,
                observed_at="2026-02-01T09:00:00+00:00",
                value=b,
                text=text(b),
            ),
            event(
                event_id="e3",
                ingestion_index=3,
                observed_at="2026-03-01T09:00:00+00:00",
                value=c,
                text=text(c),
            ),
        ]

        operations = [
            "add_initial",
            "supersede_state",
            "supersede_state",
        ]

        current_gold = c
        historical_gold = [b]
        history_values_expected = [a, b]
        stale_values = [a, b]

    elif pattern == "out_of_order":
        events = [
            event(
                event_id="e1",
                ingestion_index=1,
                observed_at="2026-01-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
            event(
                event_id="e2",
                ingestion_index=2,
                observed_at="2026-03-01T09:00:00+00:00",
                value=b,
                text=text(b),
            ),
            event(
                event_id="e3",
                ingestion_index=3,
                observed_at="2026-02-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
        ]

        operations = [
            "add_initial",
            "supersede_state",
            "reject_out_of_order",
        ]

        current_gold = b
        historical_gold = [a]
        history_values_expected = [a]
        stale_values = [a]

    elif pattern == "reversion":
        events = [
            event(
                event_id="e1",
                ingestion_index=1,
                observed_at="2026-01-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
            event(
                event_id="e2",
                ingestion_index=2,
                observed_at="2026-02-01T09:00:00+00:00",
                value=b,
                text=text(b),
            ),
            event(
                event_id="e3",
                ingestion_index=3,
                observed_at="2026-03-01T09:00:00+00:00",
                value=a,
                text=text(a),
            ),
        ]

        operations = [
            "add_initial",
            "supersede_state",
            "supersede_state",
        ]

        current_gold = a
        historical_gold = [b]
        history_values_expected = [a, b]
        stale_values = [b]

    else:
        raise ValueError(
            f"Unknown pattern: {pattern}"
        )

    return {
        "events": events,
        "expected_operations": operations,
        "current_gold": current_gold,
        "historical_gold": historical_gold,
        "history_values_expected": history_values_expected,
        "stale_values": stale_values,
    }


def build_cases() -> list[dict]:
    cases = []

    for state_index, (
        state_key,
        spec,
    ) in enumerate(
        STATE_SPECS.items(),
        start=1,
    ):
        for pattern_index, pattern in enumerate(
            PATTERNS,
            start=1,
        ):
            for variant in range(4):
                pattern_data = build_pattern(
                    pattern=pattern,
                    values=spec["values"],
                    template=spec[
                        "event_templates"
                    ][variant],
                )

                case_id = (
                    f"lc_{state_index:02d}_"
                    f"{pattern_index:02d}_"
                    f"v{variant + 1}"
                )

                case = {
                    "case_id": case_id,
                    "benchmark_version": (
                        "lifecycle_benchmark_v01"
                    ),
                    "state_key": state_key,
                    "state_label": spec["label"],
                    "pattern": pattern,
                    "surface_variant": (
                        variant + 1
                    ),
                    "user_id": (
                        f"user_{case_id}"
                    ),
                    "events": pattern_data[
                        "events"
                    ],
                    "expected_operations": (
                        pattern_data[
                            "expected_operations"
                        ]
                    ),
                    "current_query": (
                        spec["current_queries"][
                            variant
                        ]
                    ),
                    "current_gold": (
                        pattern_data[
                            "current_gold"
                        ]
                    ),
                    "historical_query": (
                        spec["history_queries"][
                            variant
                        ]
                    ),
                    "historical_gold": (
                        pattern_data[
                            "historical_gold"
                        ]
                    ),
                    "history_values_expected": (
                        pattern_data[
                            "history_values_expected"
                        ]
                    ),
                    "stale_values": (
                        pattern_data[
                            "stale_values"
                        ]
                    ),
                    "evaluation": {
                        "current_state_accuracy": True,
                        "stale_memory_exposure": True,
                        "previous_state_recall": True,
                        "history_retention_recall": True,
                        "contradiction_rate": True,
                        "memory_growth": True,
                    },
                }

                cases.append(case)

    return cases


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = build_cases()

    if len(cases) != 100:
        raise RuntimeError(
            f"Expected 100 cases, got {len(cases)}"
        )

    ids = [
        case["case_id"]
        for case in cases
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate case IDs detected."
        )

    with CASES_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                )
                + "\n"
            )

    by_pattern = Counter(
        case["pattern"]
        for case in cases
    )

    by_state_key = Counter(
        case["state_key"]
        for case in cases
    )

    manifest = {
        "benchmark": (
            "Controlled Lifecycle Benchmark"
        ),
        "version": (
            "lifecycle_benchmark_v01"
        ),
        "case_count": len(cases),
        "state_key_count": len(
            STATE_SPECS
        ),
        "pattern_count": len(
            PATTERNS
        ),
        "surface_variants": 4,
        "state_keys": list(
            STATE_SPECS
        ),
        "patterns": list(
            PATTERNS
        ),
        "cases_by_pattern": dict(
            by_pattern
        ),
        "cases_by_state_key": dict(
            by_state_key
        ),
        "primary_metrics": [
            "current_state_accuracy",
            "stale_memory_exposure_rate",
            "previous_state_recall",
            "contradiction_rate",
        ],
        "secondary_metrics": [
            "history_retention_recall",
            "memory_growth",
        ],
        "design_notes": [
            (
                "Historical is treated as a "
                "query mode rather than a "
                "lifecycle transition pattern."
            ),
            (
                "All systems receive the same "
                "event content and timestamps."
            ),
            (
                "Out-of-order cases distinguish "
                "ingestion order from observed time."
            ),
            (
                "The benchmark is controlled and "
                "does not claim to represent the "
                "full LoCoMo distribution."
            ),
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = """# Controlled Lifecycle Benchmark v0.1

This benchmark isolates memory-lifecycle behaviour from broad
open-domain conversational QA.

## Design

The benchmark contains 100 cases:

- 5 state keys;
- 5 lifecycle transition patterns;
- 4 linguistic surface variants.

## Lifecycle patterns

1. replacement: A -> B
2. duplicate: A -> A
3. multi_step: A -> B -> C
4. out_of_order: A(t1) -> B(t3) -> A(t2)
5. reversion: A -> B -> A

Historical retrieval is treated as a query mode, not as a lifecycle
transition pattern.

## Primary metrics

- Current-State Accuracy
- Stale-Memory Exposure Rate
- Previous-State Recall
- Contradiction Rate
- History-Retention Recall

## Secondary metric

- Memory Growth

## Experimental role

This benchmark is intended for controlled comparison between
C3-Lifecycle and an external memory-formation baseline such as the
pinned Mem0 OSS infer=True configuration.

It is a mechanism-isolation benchmark and does not replace LoCoMo or
other long-term conversational-memory benchmarks.
"""

    README_PATH.write_text(
        readme,
        encoding="utf-8",
    )

    print(
        "Benchmark:",
        manifest["version"],
    )
    print(
        "Cases:",
        manifest["case_count"],
    )
    print(
        "Patterns:",
        manifest["cases_by_pattern"],
    )
    print(
        "State keys:",
        manifest["cases_by_state_key"],
    )
    print(
        "Output:",
        OUT,
    )


if __name__ == "__main__":
    main()
