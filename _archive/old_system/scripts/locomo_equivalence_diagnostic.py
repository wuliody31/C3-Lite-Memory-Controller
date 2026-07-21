import csv
import json
import re
from pathlib import Path
from statistics import mean


PRED_PATH = Path("outputs/locomo_one_conv_smoke_fixed_adapter/predictions.jsonl")
IMPORT_DIR = Path("../LoCoMo_C3/neo4j_import")
REL_PATH = IMPORT_DIR / "rel_memory_relations.csv"

METHOD = "c3_lite_controller"
WINDOW = 2


def parse_dialogue_turn(memory_id: str):
    """
    e_locomo_conv_26_D1_3 -> (1, 3)
    """
    m = re.search(r"_D(\d+)_(\d+)$", memory_id)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def read_semantic_sources():
    """
    semantic observation id -> source episodic memory id
    """
    mapping = {}

    with REL_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("relation") == "DERIVED_FROM":
                mapping[row["source_id"]] = row["target_id"]

    return mapping


def source_of(memory_id: str, semantic_sources: dict):
    """
    If semantic memory, map it back to its source dialogue memory.
    If episodic memory, keep itself.
    """
    return semantic_sources.get(memory_id, memory_id)


def is_nearby(source_a: str, source_b: str, window: int = WINDOW) -> bool:
    """
    True if two source dialogue memories are in the same session
    and their turn distance is within the given window.
    """
    a = parse_dialogue_turn(source_a)
    b = parse_dialogue_turn(source_b)

    if not a or not b:
        return False

    session_a, turn_a = a
    session_b, turn_b = b

    return session_a == session_b and abs(turn_a - turn_b) <= window


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    semantic_sources = read_semantic_sources()

    rows = [
        json.loads(line)
        for line in PRED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = [r for r in rows if r["method"] == METHOD]

    strict_recalls = []
    strict_precisions = []

    source_recalls = []
    source_precisions = []

    nearby_recalls = []
    nearby_precisions = []

    print(f"Method: {METHOD}")
    print(f"Window for nearby evidence: +/- {WINDOW} turns")

    for r in rows:
        qid = r["question_id"]
        question = r["question"]

        gold_ids = list(r.get("supporting_memory_ids", []))
        used_ids = list(r.get("used_memory_ids", []))

        strict_overlap = set(gold_ids) & set(used_ids)

        gold_sources = [source_of(mid, semantic_sources) for mid in gold_ids]
        used_sources = [source_of(mid, semantic_sources) for mid in used_ids]

        gold_source_set = set(gold_sources)
        used_source_set = set(used_sources)

        source_overlap = gold_source_set & used_source_set

        nearby_gold_matched = set()
        nearby_used_matched = set()

        for g in gold_source_set:
            for u in used_source_set:
                if is_nearby(g, u, WINDOW):
                    nearby_gold_matched.add(g)
                    nearby_used_matched.add(u)

        strict_recall = safe_div(len(strict_overlap), len(set(gold_ids)))
        strict_precision = safe_div(len(strict_overlap), len(set(used_ids)))

        source_recall = safe_div(len(source_overlap), len(gold_source_set))
        source_precision = safe_div(len(source_overlap), len(used_source_set))

        nearby_recall = safe_div(len(nearby_gold_matched), len(gold_source_set))
        nearby_precision = safe_div(len(nearby_used_matched), len(used_source_set))

        strict_recalls.append(strict_recall)
        strict_precisions.append(strict_precision)

        source_recalls.append(source_recall)
        source_precisions.append(source_precision)

        nearby_recalls.append(nearby_recall)
        nearby_precisions.append(nearby_precision)

        print("\n" + "=" * 100)
        print("ID:", qid)
        print("Q:", question)

        print("Gold IDs:", gold_ids)
        print("Used IDs:", used_ids)

        print("Gold sources:", sorted(gold_source_set))
        print("Used sources:", sorted(used_source_set))

        print("Strict overlap:", sorted(strict_overlap))
        print("Source-equivalent overlap:", sorted(source_overlap))
        print("Nearby gold matched:", sorted(nearby_gold_matched))
        print("Nearby used matched:", sorted(nearby_used_matched))

        print(
            "Scores:",
            {
                "strict_recall": round(strict_recall, 4),
                "strict_precision": round(strict_precision, 4),
                "source_recall": round(source_recall, 4),
                "source_precision": round(source_precision, 4),
                "nearby_recall": round(nearby_recall, 4),
                "nearby_precision": round(nearby_precision, 4),
            },
        )

    print("\n" + "=" * 100)
    print("MACRO SUMMARY")
    print(
        {
            "strict_recall": round(mean(strict_recalls), 4),
            "strict_precision": round(mean(strict_precisions), 4),
            "source_equiv_recall": round(mean(source_recalls), 4),
            "source_equiv_precision": round(mean(source_precisions), 4),
            "nearby_equiv_recall": round(mean(nearby_recalls), 4),
            "nearby_equiv_precision": round(mean(nearby_precisions), 4),
        }
    )


if __name__ == "__main__":
    main()