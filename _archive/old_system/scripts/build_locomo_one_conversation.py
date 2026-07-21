import csv
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict


RAW_PATH = Path("../LoCoMo_C3/data_raw/locomo10.json")
OUT_ROOT = Path("../LoCoMo_C3")
OUT_RAW = OUT_ROOT / "data_raw"
OUT_EVAL = OUT_ROOT / "data_eval"
OUT_IMPORT = OUT_ROOT / "neo4j_import"

DATASET_A_IMPORT = Path("../Dataset_A_v0_2/neo4j_import")
DATASET_A_CYPHER = DATASET_A_IMPORT / "neo4j_full_import.cypher"

MAX_QA = 5


def safe_id(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "unknown"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path, "rows:", len(rows))


def dialogue_id_to_memory_id(sample_safe: str, dia_id: str) -> str:
    return f"e_locomo_{sample_safe}_{safe_id(dia_id)}"


def observation_id(sample_safe: str, index: int) -> str:
    return f"s_locomo_{sample_safe}_obs_{index:04d}"


def entity_id(name: str) -> str:
    return f"ent_{safe_id(name)}"


def session_number_from_key(session_key: str) -> int:
    # session_12 -> 12
    return int(session_key.split("_")[1])


def normalise_evidence_ids(value) -> list[str]:
    """
    LoCoMo observations usually store one evidence id like "D1:3".
    This helper also handles list-style evidence just in case.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    return [str(value)]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    sample = data[0]

    sample_id = sample["sample_id"]
    sample_safe = safe_id(sample_id)

    user_id = f"locomo_{sample_safe}"

    conversation = sample["conversation"]
    speaker_a = conversation.get("speaker_a", "speaker_a")
    speaker_b = conversation.get("speaker_b", "speaker_b")

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_IMPORT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    users = [
        {
            "user_id": user_id,
            "name": sample_id,
            "domain": "LoCoMo long-term conversational memory benchmark",
            "long_term_goal": (
                "Evaluate long-term conversational memory retrieval "
                "and question answering."
            ),
            "stable_preferences": (
                f"Conversation between {speaker_a} and {speaker_b}"
            ),
            "notes": (
                "Converted from LoCoMo. Dialogue turns are episodic "
                "memories; observations are semantic memories."
            ),
        }
    ]

    # ------------------------------------------------------------------
    # Sessions and episodic memories
    # ------------------------------------------------------------------
    sessions = []
    episodic_events = []
    rel_user_has_session = []
    rel_session_has_event = []
    rel_user_has_event = []
    rel_event_mentions_entity = []

    entities = {}

    # D1:3 -> e_locomo_conv_26_D1_3
    dia_to_memory_id = {}

    # D1:3 -> [s_locomo_conv_26_obs_0001, ...]
    # This is used to expand LoCoMo gold evidence from dialogue turns
    # to equivalent semantic observation memories.
    dia_to_semantic_ids = defaultdict(list)

    session_keys = [
        key for key in conversation.keys()
        if key.startswith("session_")
        and not key.endswith("_date_time")
    ]

    session_keys = sorted(
        session_keys,
        key=session_number_from_key,
    )

    for session_key in session_keys:
        session_num = session_number_from_key(session_key)
        session_id = f"{user_id}_s{session_num:02d}"
        date_key = f"{session_key}_date_time"
        date = conversation.get(date_key, "")

        turns = conversation.get(session_key, [])

        sessions.append({
            "session_id": session_id,
            "user_id": user_id,
            "date": date,
            "title": f"LoCoMo {sample_id} {session_key}",
            "turn_count": str(len(turns)),
        })

        rel_user_has_session.append({
            "user_id": user_id,
            "session_id": session_id,
            "relation": "HAS_SESSION",
        })

        for turn_index, turn in enumerate(turns, start=1):
            dia_id = turn.get("dia_id", f"D{session_num}:{turn_index}")
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")

            memory_id = dialogue_id_to_memory_id(sample_safe, dia_id)
            dia_to_memory_id[dia_id] = memory_id

            event_text = f"{speaker}: {text}"

            episodic_events.append({
                "memory_id": memory_id,
                "user_id": user_id,
                "date": date,
                "event": event_text,
                "entities": speaker,
                "source_session": session_id,
                "importance": "3",
                "memory_type": "episodic",
            })

            rel_session_has_event.append({
                "session_id": session_id,
                "memory_id": memory_id,
                "relation": "CONTAINS_EVENT",
            })

            rel_user_has_event.append({
                "user_id": user_id,
                "memory_id": memory_id,
                "relation": "HAS_EPISODIC_MEMORY",
            })

            if speaker:
                eid = entity_id(speaker)
                entities[eid] = {
                    "entity_id": eid,
                    "name": speaker,
                    "label_hint": "Person",
                }

                rel_event_mentions_entity.append({
                    "memory_id": memory_id,
                    "entity_id": eid,
                    "entity_name": speaker,
                    "relation": "MENTIONS",
                })

    # ------------------------------------------------------------------
    # Semantic memories from LoCoMo observations
    # ------------------------------------------------------------------
    semantic_facts = []
    rel_user_has_semantic_fact = []
    rel_semantic_edges = []
    rel_memory_relations = []

    observations = sample.get("observation", {})

    obs_index = 1

    for obs_key, obs_value in observations.items():
        session_key = obs_key.replace("_observation", "")
        session_num = session_number_from_key(session_key)
        session_id = f"{user_id}_s{session_num:02d}"
        date = conversation.get(f"{session_key}_date_time", "")

        if not isinstance(obs_value, dict):
            continue

        for subject, facts in obs_value.items():
            subject_entity = entity_id(subject)

            entities[subject_entity] = {
                "entity_id": subject_entity,
                "name": subject,
                "label_hint": "Person",
            }

            for item in facts:
                if not isinstance(item, list) or len(item) < 2:
                    continue

                fact_text = str(item[0])
                evidence_dia_ids = normalise_evidence_ids(item[1])

                triple_id = observation_id(sample_safe, obs_index)
                obs_index += 1

                object_text = fact_text
                object_entity = entity_id(object_text)

                entities[object_entity] = {
                    "entity_id": object_entity,
                    "name": object_text,
                    "label_hint": "Observation",
                }

                semantic_facts.append({
                    "triple_id": triple_id,
                    "user_id": user_id,
                    "subject": subject,
                    "relation": "observed_fact",
                    "object": object_text,
                    "confidence": "0.90",
                    "source": session_id,
                    "last_updated": date,
                    "status": "active",
                    "memory_type": "semantic",
                })

                rel_user_has_semantic_fact.append({
                    "user_id": user_id,
                    "triple_id": triple_id,
                    "relation": "HAS_SEMANTIC_FACT",
                })

                rel_semantic_edges.append({
                    "triple_id": triple_id,
                    "user_id": user_id,
                    "subject_entity_id": subject_entity,
                    "subject": subject,
                    "object_entity_id": object_entity,
                    "object": object_text,
                    "relation": "observed_fact",
                    "confidence": "0.90",
                    "source": session_id,
                    "last_updated": date,
                    "status": "active",
                })

                # Link each semantic observation to its source dialogue turn.
                # Also store this mapping so eval questions can treat
                # dialogue-turn evidence and derived semantic evidence as equivalent.
                for evidence_dia_id in evidence_dia_ids:
                    source_memory_id = dia_to_memory_id.get(evidence_dia_id)

                    if source_memory_id:
                        dia_to_semantic_ids[evidence_dia_id].append(triple_id)

                        rel_memory_relations.append({
                            "source_id": triple_id,
                            "target_id": source_memory_id,
                            "relation": "DERIVED_FROM",
                            "reason": (
                                f"LoCoMo observation derived from dialogue "
                                f"{evidence_dia_id}."
                            ),
                        })

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    entities_rows = list(entities.values())

    # ------------------------------------------------------------------
    # Evaluation questions
    # ------------------------------------------------------------------
    eval_rows = []
    evidence_equivalence_map = {}

    for i, qa in enumerate(sample["qa"][:MAX_QA], start=1):
        evidence_dialogue_ids = qa.get("evidence", [])

        supporting_memory_ids = []

        for dia_id in evidence_dialogue_ids:
            # 1. Add original dialogue-turn episodic evidence.
            episodic_id = dia_to_memory_id.get(dia_id)

            if episodic_id and episodic_id not in supporting_memory_ids:
                supporting_memory_ids.append(episodic_id)

            # 2. Add semantic observations derived from this dialogue turn.
            for semantic_id in dia_to_semantic_ids.get(dia_id, []):
                if semantic_id not in supporting_memory_ids:
                    supporting_memory_ids.append(semantic_id)

        question_id = f"q_locomo_{sample_safe}_{i:04d}"
        category = qa.get("category")

        evidence_equivalence_map[question_id] = {
            "locomo_evidence_dialogue_ids": evidence_dialogue_ids,
            "supporting_memory_ids": supporting_memory_ids,
        }

        eval_rows.append({
            "question_id": question_id,
            "user_id": user_id,
            "question_type": f"locomo_category_{category}",
            "question": qa.get("question", ""),
            "gold_answer": str(qa.get("answer", "")),
            "expected_route": [
                "neo4j_episodic_graph",
                "neo4j_semantic_graph",
            ],
            "required_memory": [
                "neo4j_episodic_graph",
                "neo4j_semantic_graph",
            ],
            "supporting_memory_ids": supporting_memory_ids,
            "answer_should_include": [],
            "answer_should_not_include": [],
            "should_abstain": False,
            "expected_outdated_memory_ids": [],
            "conflict_type": "none",
            "locomo_evidence_dialogue_ids": evidence_dialogue_ids,
            "locomo_category": category,
        })

    eval_path = OUT_EVAL / "eval_questions.jsonl"

    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", eval_path, "rows:", len(eval_rows))

    equivalence_path = OUT_EVAL / "evidence_equivalence_map.json"
    equivalence_path.write_text(
        json.dumps(evidence_equivalence_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved:", equivalence_path)

    # ------------------------------------------------------------------
    # Procedural memory
    # IMPORTANT:
    # The procedural matcher expects condition/action fields.
    # Do not use title/rule here.
    # ------------------------------------------------------------------
    procedural_rules = [
        {
            "rule_id": f"p_{user_id}_001",
            "user_id": user_id,
            "name": "Grounded LoCoMo answering",
            "condition": (
                "The user asks a LoCoMo question about the stored conversation."
            ),
            "action": (
                "Answer using only the stored conversation turns and observations. "
                "Do not use QA gold answers as memory evidence."
            ),
            "priority": 5,
            "status": "active",
            "memory_type": "procedural",
        },
        {
            "rule_id": f"p_{user_id}_002",
            "user_id": user_id,
            "name": "Uncertainty policy",
            "condition": (
                "The stored conversation evidence is insufficient to answer confidently."
            ),
            "action": (
                "State that there is not enough stored evidence instead of guessing."
            ),
            "priority": 5,
            "status": "active",
            "memory_type": "procedural",
        },
    ]

    procedural_path = OUT_RAW / "procedural_memory.json"

    procedural_path.write_text(
        json.dumps(procedural_rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved:", procedural_path)

    # ------------------------------------------------------------------
    # Write Neo4j CSVs
    # ------------------------------------------------------------------
    write_csv(
        OUT_IMPORT / "nodes_users.csv",
        ["user_id", "name", "domain", "long_term_goal", "stable_preferences", "notes"],
        users,
    )

    write_csv(
        OUT_IMPORT / "nodes_sessions.csv",
        ["session_id", "user_id", "date", "title", "turn_count"],
        sessions,
    )

    write_csv(
        OUT_IMPORT / "nodes_episodic_events.csv",
        [
            "memory_id",
            "user_id",
            "date",
            "event",
            "entities",
            "source_session",
            "importance",
            "memory_type",
        ],
        episodic_events,
    )

    write_csv(
        OUT_IMPORT / "nodes_semantic_facts.csv",
        [
            "triple_id",
            "user_id",
            "subject",
            "relation",
            "object",
            "confidence",
            "source",
            "last_updated",
            "status",
            "memory_type",
        ],
        semantic_facts,
    )

    write_csv(
        OUT_IMPORT / "nodes_entities.csv",
        ["entity_id", "name", "label_hint"],
        entities_rows,
    )

    write_csv(
        OUT_IMPORT / "rel_user_has_session.csv",
        ["user_id", "session_id", "relation"],
        rel_user_has_session,
    )

    write_csv(
        OUT_IMPORT / "rel_session_has_event.csv",
        ["session_id", "memory_id", "relation"],
        rel_session_has_event,
    )

    write_csv(
        OUT_IMPORT / "rel_user_has_event.csv",
        ["user_id", "memory_id", "relation"],
        rel_user_has_event,
    )

    write_csv(
        OUT_IMPORT / "rel_user_has_semantic_fact.csv",
        ["user_id", "triple_id", "relation"],
        rel_user_has_semantic_fact,
    )

    write_csv(
        OUT_IMPORT / "rel_event_mentions_entity.csv",
        ["memory_id", "entity_id", "entity_name", "relation"],
        rel_event_mentions_entity,
    )

    write_csv(
        OUT_IMPORT / "rel_semantic_edges.csv",
        [
            "triple_id",
            "user_id",
            "subject_entity_id",
            "subject",
            "object_entity_id",
            "object",
            "relation",
            "confidence",
            "source",
            "last_updated",
            "status",
        ],
        rel_semantic_edges,
    )

    write_csv(
        OUT_IMPORT / "rel_memory_relations.csv",
        ["source_id", "target_id", "relation", "reason"],
        rel_memory_relations,
    )

    # Reuse the existing Cypher import script because CSV headers match Dataset A.
    if DATASET_A_CYPHER.exists():
        shutil.copy(
            DATASET_A_CYPHER,
            OUT_IMPORT / "neo4j_full_import.cypher",
        )
        print("Copied import cypher:", OUT_IMPORT / "neo4j_full_import.cypher")
    else:
        print("WARNING: Dataset A import cypher not found:", DATASET_A_CYPHER)

    print("\nDone.")
    print("User ID:", user_id)
    print("Sessions:", len(sessions))
    print("Episodic memories:", len(episodic_events))
    print("Semantic facts:", len(semantic_facts))
    print("Entities:", len(entities_rows))
    print("Eval questions:", len(eval_rows))


if __name__ == "__main__":
    main()