from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser


QUESTION_FIELDS = [
    "question_id",
    "user_id",
    "question_type",
    "question",
    "supporting_memory_ids",
    "should_abstain",
    "expected_outdated_memory_ids",
    "conflict_type",
    "gold_answer",
    "expected_memory_types",
    "summary_expected_memory_types",
    "route_metric_applicable",
    "route_label_source",
    "route_label_note",
]


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalise_answer(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float, bool)):
        return str(value)

    if isinstance(value, list):
        return "; ".join(
            normalise_answer(item)
            for item in value
            if normalise_answer(item)
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def normalise_references(value: Any) -> list[str]:
    if value in (None, ""):
        return []

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, (list, tuple, set)):
        output = []

        for item in value:
            output.extend(
                normalise_references(item)
            )

        return output

    return [str(value).strip()]


def parse_timestamp(value: Any) -> str | None:
    text = str(value or "").strip()

    if not text:
        return None

    text = re.sub(
        r"\s+on\s+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    parsed = date_parser.parse(
        text,
        fuzzy=True,
    )

    return parsed.isoformat()


def session_number(key: str) -> int:
    match = re.fullmatch(
        r"session_(\d+)",
        key,
    )

    if not match:
        raise ValueError(
            f"Invalid session key: {key}"
        )

    return int(match.group(1))


def session_keys(
    conversation: dict[str, Any],
) -> list[str]:
    keys = [
        key
        for key, value in conversation.items()
        if (
            re.fullmatch(
                r"session_\d+",
                key,
            )
            and isinstance(value, list)
        )
    ]

    return sorted(
        keys,
        key=session_number,
    )


def episodic_id(
    user_id: str,
    raw_dialogue_id: str,
) -> str:
    return (
        f"e_{user_id}_"
        f"{slug(raw_dialogue_id)}"
    )


def observation_entry(
    value: Any,
) -> tuple[str, list[str]]:
    if isinstance(value, (list, tuple)):
        text = (
            normalise_answer(value[0])
            if value
            else ""
        )

        references = (
            normalise_references(value[1])
            if len(value) >= 2
            else []
        )

        return text, references

    if isinstance(value, dict):
        text = normalise_answer(
            value.get("text")
            or value.get("observation")
            or value.get("summary")
        )

        references = normalise_references(
            value.get("evidence")
            or value.get("dia_id")
            or value.get("source")
        )

        return text, references

    return normalise_answer(value), []


def stable_hash(
    text: str,
    seed: int,
) -> str:
    return hashlib.sha256(
        f"{seed}|{text}".encode("utf-8")
    ).hexdigest()


def balanced_subset(
    rows: list[dict[str, str]],
    size: int,
    seed: int,
) -> list[dict[str, str]]:
    if size <= 0:
        return []

    categories = sorted({
        row["question_type"]
        for row in rows
    })

    if not categories:
        return []

    base = size // len(categories)
    remainder = size % len(categories)

    selected: list[dict[str, str]] = []

    for category_index, category in enumerate(
        categories
    ):
        quota = (
            base
            + int(category_index < remainder)
        )

        candidates = [
            row
            for row in rows
            if row["question_type"] == category
        ]

        by_user: dict[
            str,
            list[dict[str, str]],
        ] = defaultdict(list)

        for row in candidates:
            by_user[row["user_id"]].append(
                row
            )

        for user_id in by_user:
            by_user[user_id].sort(
                key=lambda row: stable_hash(
                    row["question_id"],
                    seed,
                )
            )

        users = sorted(
            by_user,
            key=lambda user_id: stable_hash(
                user_id,
                seed,
            ),
        )

        category_selected: list[
            dict[str, str]
        ] = []

        while len(category_selected) < quota:
            made_progress = False

            for user_id in users:
                if not by_user[user_id]:
                    continue

                category_selected.append(
                    by_user[user_id].pop(0)
                )
                made_progress = True

                if (
                    len(category_selected)
                    >= quota
                ):
                    break

            if not made_progress:
                break

        if len(category_selected) != quota:
            raise RuntimeError(
                f"Category {category} supplied "
                f"{len(category_selected)}/{quota} "
                "requested rows."
            )

        selected.extend(category_selected)

    selected.sort(
        key=lambda row: row["question_id"]
    )

    if len(selected) != size:
        raise RuntimeError(
            f"Expected subset size {size}, "
            f"found {len(selected)}."
        )

    return selected


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def prepare(
    *,
    source: Path,
    output_dir: Path,
    include_observations: bool,
    smoke_size: int,
    pilot_size: int,
    seed: int,
    strict: bool,
) -> dict[str, Any]:
    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise TypeError(
            "LoCoMo top level must be a list."
        )

    memories: list[dict[str, Any]] = []
    questions: list[dict[str, str]] = []
    mapping_audit: list[dict[str, Any]] = []

    global_memory_ids: set[str] = set()
    raw_dialogue_ids: list[str] = []

    unresolved_gold: list[dict[str, str]] = []
    unresolved_observations: list[
        dict[str, str]
    ] = []

    episodic_count = 0
    semantic_count = 0

    for sample in data:
        sample_id = str(
            sample["sample_id"]
        )
        user_id = (
            f"locomo_{slug(sample_id)}"
        )

        conversation = sample["conversation"]

        if not isinstance(
            conversation,
            dict,
        ):
            raise TypeError(
                f"Conversation {sample_id} "
                "must be a dictionary."
            )

        dialogue_map: dict[str, str] = {}
        session_timestamps: dict[
            str,
            str | None,
        ] = {}

        sample_turns = 0

        for session_key in session_keys(
            conversation
        ):
            number = session_number(
                session_key
            )

            timestamp = parse_timestamp(
                conversation.get(
                    f"{session_key}_date_time"
                )
            )

            session_timestamps[
                session_key
            ] = timestamp

            for turn_index, turn in enumerate(
                conversation[session_key],
                start=1,
            ):
                raw_id = str(
                    turn.get("dia_id", "")
                ).strip()

                if not raw_id:
                    raise ValueError(
                        f"{sample_id} {session_key} "
                        f"turn {turn_index} has no dia_id."
                    )

                if raw_id in dialogue_map:
                    raise ValueError(
                        f"Duplicate dia_id {raw_id} "
                        f"inside {sample_id}."
                    )

                memory_id = episodic_id(
                    user_id,
                    raw_id,
                )

                if memory_id in global_memory_ids:
                    raise ValueError(
                        f"Duplicate namespaced memory ID: "
                        f"{memory_id}"
                    )

                speaker = str(
                    turn.get("speaker", "")
                ).strip()

                text = str(
                    turn.get("text", "")
                ).strip()

                dialogue_map[raw_id] = memory_id
                global_memory_ids.add(memory_id)
                raw_dialogue_ids.append(raw_id)

                memories.append({
                    "memory_id": memory_id,
                    "memory_type": "episodic",
                    "text": (
                        f"{speaker}: {text}"
                        if speaker
                        else text
                    ),
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "status": "current",
                    "confidence": 1.0,
                    "importance": 0.5,
                    "authority": (
                        "official_locomo_dialogue"
                    ),
                    "source_ids": [memory_id],
                    "relations": [],
                    "metadata": {
                        "dataset": "LoCoMo",
                        "sample_id": sample_id,
                        "raw_dia_id": raw_id,
                        "source_session": (
                            session_key
                        ),
                        "session_number": number,
                        "speaker": speaker,
                        "turn_index": turn_index,
                        "entities": (
                            [speaker]
                            if speaker
                            else []
                        ),
                    },
                })

                episodic_count += 1
                sample_turns += 1

        sample_semantic = 0

        if include_observations:
            observations = sample.get(
                "observation",
                {},
            )

            if not isinstance(
                observations,
                dict,
            ):
                raise TypeError(
                    f"Observation {sample_id} "
                    "must be a dictionary."
                )

            for observation_key in sorted(
                observations
            ):
                match = re.fullmatch(
                    r"session_(\d+)_observation",
                    observation_key,
                )

                if not match:
                    continue

                number = int(
                    match.group(1)
                )
                source_session = (
                    f"session_{number}"
                )

                timestamp = (
                    session_timestamps.get(
                        source_session
                    )
                )

                by_speaker = observations[
                    observation_key
                ]

                if not isinstance(
                    by_speaker,
                    dict,
                ):
                    continue

                for speaker in sorted(
                    by_speaker
                ):
                    entries = by_speaker[
                        speaker
                    ]

                    if not isinstance(
                        entries,
                        list,
                    ):
                        continue

                    for item_index, entry in enumerate(
                        entries,
                        start=1,
                    ):
                        text, raw_sources = (
                            observation_entry(
                                entry
                            )
                        )

                        if not text:
                            continue

                        mapped_sources = []

                        for raw_source in raw_sources:
                            mapped = dialogue_map.get(
                                raw_source
                            )

                            if mapped is None:
                                unresolved_observations.append({
                                    "sample_id": sample_id,
                                    "observation_key": (
                                        observation_key
                                    ),
                                    "raw_source": raw_source,
                                })
                                continue

                            mapped_sources.append(
                                mapped
                            )

                        memory_id = (
                            f"s_{user_id}_obs_"
                            f"s{number:02d}_"
                            f"{slug(speaker)}_"
                            f"{item_index:03d}"
                        )

                        if (
                            memory_id
                            in global_memory_ids
                        ):
                            raise ValueError(
                                "Duplicate semantic ID: "
                                f"{memory_id}"
                            )

                        global_memory_ids.add(
                            memory_id
                        )

                        memories.append({
                            "memory_id": memory_id,
                            "memory_type": (
                                "semantic"
                            ),
                            "text": text,
                            "user_id": user_id,
                            "timestamp": timestamp,
                            "status": "unknown",
                            "confidence": 1.0,
                            "importance": 0.7,
                            "authority": (
                                "official_locomo_observation"
                            ),
                            "source_ids": (
                                mapped_sources
                            ),
                            "relations": [],
                            "metadata": {
                                "dataset": "LoCoMo",
                                "sample_id": sample_id,
                                "source_session": (
                                    source_session
                                ),
                                "session_number": (
                                    number
                                ),
                                "speaker": speaker,
                                "raw_source_ids": (
                                    raw_sources
                                ),
                                "entities": [
                                    speaker
                                ],
                                "source_kind": (
                                    "official_observation"
                                ),
                            },
                        })

                        semantic_count += 1
                        sample_semantic += 1

        sample_questions = 0
        sample_answerable = 0
        sample_without_evidence = 0

        for qa_index, qa in enumerate(
            sample.get("qa", []),
            start=1,
        ):
            question_id = (
                f"q_{user_id}_"
                f"{qa_index:04d}"
            )

            raw_evidence = (
                normalise_references(
                    qa.get("evidence")
                )
            )

            mapped_evidence = []

            for raw_id in raw_evidence:
                mapped = dialogue_map.get(
                    raw_id
                )

                if mapped is None:
                    unresolved_gold.append({
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "raw_evidence_id": raw_id,
                    })
                    continue

                mapped_evidence.append(
                    mapped
                )

            answer = normalise_answer(
                qa.get("answer")
            )
            should_abstain = not bool(
                answer.strip()
            )

            if answer:
                sample_answerable += 1

            if not raw_evidence:
                sample_without_evidence += 1

            category = str(
                qa.get("category", "unknown")
            )

            questions.append({
                "question_id": question_id,
                "user_id": user_id,
                "question_type": (
                    f"locomo_category_{category}"
                ),
                "question": str(
                    qa.get("question", "")
                ).strip(),
                "supporting_memory_ids": (
                    ";".join(mapped_evidence)
                ),
                "should_abstain": str(
                    should_abstain
                ),
                "expected_outdated_memory_ids": "",
                "conflict_type": "none",
                "gold_answer": answer,
                "expected_memory_types": "",
                "summary_expected_memory_types": "",
                "route_metric_applicable": (
                    "False"
                ),
                "route_label_source": (
                    "not_available_external_dataset"
                ),
                "route_label_note": (
                    "LoCoMo supplies dialogue evidence "
                    "IDs but no memory-type route labels."
                ),
            })

            sample_questions += 1

        mapping_audit.append({
            "sample_id": sample_id,
            "user_id": user_id,
            "session_count": len(
                session_keys(conversation)
            ),
            "dialogue_turns": sample_turns,
            "semantic_observations": (
                sample_semantic
            ),
            "qa_total": sample_questions,
            "qa_answerable": (
                sample_answerable
            ),
            "qa_unanswerable": (
                sample_questions
                - sample_answerable
            ),
            "qa_without_evidence": (
                sample_without_evidence
            ),
        })

    question_ids = [
        row["question_id"]
        for row in questions
    ]

    if (
        len(question_ids)
        != len(set(question_ids))
    ):
        raise RuntimeError(
            "Duplicate question IDs detected."
        )

    answerable_rows = [
        row
        for row in questions
        if row["should_abstain"] == "False"
    ]

    smoke_rows = balanced_subset(
        questions,
        smoke_size,
        seed,
    )

    pilot_rows = balanced_subset(
        questions,
        pilot_size,
        seed,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output_dir
        / "eval_questions_locomo_all.csv",
        QUESTION_FIELDS,
        questions,
    )

    write_csv(
        output_dir
        / "eval_questions_locomo_answerable.csv",
        QUESTION_FIELDS,
        answerable_rows,
    )

    write_csv(
        output_dir
        / "eval_questions_locomo_smoke50.csv",
        QUESTION_FIELDS,
        smoke_rows,
    )

    write_csv(
        output_dir
        / "eval_questions_locomo_pilot200.csv",
        QUESTION_FIELDS,
        pilot_rows,
    )

    write_json(
        output_dir / "memories_locomo.json",
        {"memories": memories},
    )

    write_json(
        output_dir / "procedures_locomo.json",
        {"procedures": []},
    )

    write_csv(
        output_dir / "mapping_audit.csv",
        list(mapping_audit[0].keys()),
        mapping_audit,
    )

    category_rows = []

    for category in sorted(
        {
            row["question_type"]
            for row in questions
        }
    ):
        category_questions = [
            row
            for row in questions
            if row["question_type"]
            == category
        ]

        category_rows.append({
            "question_type": category,
            "total": len(
                category_questions
            ),
            "answerable": sum(
                row["should_abstain"]
                == "False"
                for row in category_questions
            ),
            "unanswerable": sum(
                row["should_abstain"]
                == "True"
                for row in category_questions
            ),
            "with_evidence": sum(
                bool(
                    row[
                        "supporting_memory_ids"
                    ]
                )
                for row in category_questions
            ),
        })

    write_csv(
        output_dir / "category_counts.csv",
        list(category_rows[0].keys()),
        category_rows,
    )

    source_commit_path = (
        source.parent.parent
        / "SOURCE_COMMIT.txt"
    )

    source_commit = (
        source_commit_path.read_text(
            encoding="utf-8"
        ).strip()
        if source_commit_path.exists()
        else None
    )

    manifest = {
        "adapter_version": "locomo_adapter_v01",
        "source": str(source),
        "source_sha256": sha256_file(
            source
        ),
        "source_commit": source_commit,
        "include_official_observations": (
            include_observations
        ),
        "seed": seed,
        "counts": {
            "samples": len(data),
            "questions_total": len(
                questions
            ),
            "questions_answerable": len(
                answerable_rows
            ),
            "questions_unanswerable": (
                len(questions)
                - len(answerable_rows)
            ),
            "questions_without_evidence": sum(
                not row[
                    "supporting_memory_ids"
                ]
                for row in questions
            ),
            "smoke_questions": len(
                smoke_rows
            ),
            "pilot_questions": len(
                pilot_rows
            ),
            "episodic_memories": (
                episodic_count
            ),
            "semantic_memories": (
                semantic_count
            ),
            "procedural_memories": 0,
            "raw_dialogue_ids_total": len(
                raw_dialogue_ids
            ),
            "raw_dialogue_ids_unique": len(
                set(raw_dialogue_ids)
            ),
            "namespaced_memory_ids_unique": (
                len(global_memory_ids)
            ),
        },
        "question_type_counts": dict(
            sorted(
                Counter(
                    row["question_type"]
                    for row in questions
                ).items()
            )
        ),
        "unresolved_gold_evidence": (
            unresolved_gold
        ),
        "unresolved_observation_sources": (
            unresolved_observations
        ),
        "evaluation_policy": {
            "answer_f1_primary_file": (
                "eval_questions_locomo_answerable.csv"
            ),
            "abstention_file": (
                "eval_questions_locomo_all.csv"
            ),
            "route_metrics": False,
        },
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    checksum_paths = sorted(
        path
        for path in output_dir.iterdir()
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        )
    )

    with (
        output_dir / "SHA256SUMS.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        for path in checksum_paths:
            handle.write(
                f"{sha256_file(path)}  "
                f"{path.name}\n"
            )

    print("=" * 80)
    print("LOCOMO ADAPTER V0.1")
    print("=" * 80)
    print("Samples:", len(data))
    print("Questions:", len(questions))
    print(
        "Answerable:",
        len(answerable_rows),
    )
    print(
        "Unanswerable:",
        len(questions)
        - len(answerable_rows),
    )
    print(
        "Episodic memories:",
        episodic_count,
    )
    print(
        "Semantic memories:",
        semantic_count,
    )
    print(
        "Raw dialogue IDs:",
        len(raw_dialogue_ids),
    )
    print(
        "Unique raw dialogue IDs:",
        len(set(raw_dialogue_ids)),
    )
    print(
        "Unique namespaced IDs:",
        len(global_memory_ids),
    )
    print(
        "Unresolved gold IDs:",
        len(unresolved_gold),
    )
    print(
        "Unresolved observation sources:",
        len(unresolved_observations),
    )
    print("Output:", output_dir)

    if strict and (
        unresolved_gold
        or unresolved_observations
    ):
        raise RuntimeError(
            "Strict validation failed. "
            "Inspect manifest.json."
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--include-observations",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260801,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
    )

    args = parser.parse_args()

    prepare(
        source=args.source,
        output_dir=args.output_dir,
        include_observations=(
            args.include_observations
        ),
        smoke_size=args.smoke_size,
        pilot_size=args.pilot_size,
        seed=args.seed,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
