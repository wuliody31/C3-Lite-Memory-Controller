import json
import csv
from pathlib import Path
from collections import Counter


pred_path = Path("outputs/full_v01_conflict_metrics/predictions.jsonl")
out_path = Path("outputs/full_v01_conflict_metrics/conflict_error_analysis.csv")


def stringify_note(note):
    """
    Convert conflict notes into readable strings.
    Some notes are strings, while others may be dict objects.
    """
    if isinstance(note, str):
        return note

    if isinstance(note, dict):
        return json.dumps(note, ensure_ascii=False)

    return str(note)


rows = []

for line in pred_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue

    r = json.loads(line)

    if r["method"] != "c3_lite_controller":
        continue

    gold = set(r.get("expected_outdated_memory_ids", []))
    pred = set(r.get("outdated_memory_ids", []))

    if not gold and not pred:
        continue

    overlap = gold & pred
    false_positive = pred - gold
    false_negative = gold - pred

    if gold and not pred:
        error_type = "missed_conflict"
    elif pred and not gold:
        error_type = "false_conflict"
    elif overlap and false_positive and false_negative:
        error_type = "partial_overlap_fp_and_fn"
    elif overlap and false_positive:
        error_type = "partial_overlap_extra_predictions"
    elif overlap and false_negative:
        error_type = "partial_overlap_missing_gold"
    elif overlap:
        error_type = "exact_or_sufficient_match"
    else:
        error_type = "wrong_conflict_ids"

    conflict_notes = r.get("conflict_notes", [])
    conflict_notes_text = " | ".join(
        stringify_note(note)
        for note in conflict_notes
    )

    rows.append({
        "question_id": r.get("question_id"),
        "question_type": r.get("question_type"),
        "conflict_type": r.get("conflict_type"),
        "question": r.get("question"),
        "gold_outdated": ";".join(sorted(gold)),
        "pred_outdated": ";".join(sorted(pred)),
        "overlap": ";".join(sorted(overlap)),
        "false_positive": ";".join(sorted(false_positive)),
        "false_negative": ";".join(sorted(false_negative)),
        "error_type": error_type,
        "used_memory_ids": ";".join(r.get("used_memory_ids", [])),
        "conflict_notes": conflict_notes_text,
    })


out_path.parent.mkdir(parents=True, exist_ok=True)

if rows:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_path)
    print("Rows:", len(rows))
    print(Counter(r["error_type"] for r in rows))
else:
    print("No conflict-related rows found.")