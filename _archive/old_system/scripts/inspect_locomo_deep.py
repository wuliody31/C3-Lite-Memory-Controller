import json
from pathlib import Path
from collections import Counter


path = Path("../LoCoMo_C3/data_raw/locomo10.json")
data = json.loads(path.read_text(encoding="utf-8"))

sample = data[0]

print("sample_id:", sample.get("sample_id"))
print("sample keys:", list(sample.keys()))

conversation = sample["conversation"]

print("\nSpeakers:")
print("speaker_a:", conversation.get("speaker_a"))
print("speaker_b:", conversation.get("speaker_b"))

session_keys = [
    key for key in conversation.keys()
    if key.startswith("session_") and not key.endswith("_date_time")
]

session_keys = sorted(
    session_keys,
    key=lambda x: int(x.split("_")[1])
)

print("\nNumber of sessions:", len(session_keys))
print("Session keys:", session_keys[:5], "...")

for session_key in session_keys[:2]:
    date_key = f"{session_key}_date_time"

    print("\n" + "=" * 80)
    print("SESSION:", session_key)
    print("DATE:", conversation.get(date_key))

    turns = conversation.get(session_key, [])
    print("Turns type:", type(turns))
    print("Number of turns:", len(turns))

    if turns:
        print("\nFirst 3 turns:")
        for turn in turns[:3]:
            print(json.dumps(turn, ensure_ascii=False, indent=2)[:1500])

print("\n" + "=" * 80)
print("QA category distribution:")
print(Counter(item.get("category") for item in sample["qa"]))

print("\nFirst 5 QA examples:")
for qa in sample["qa"][:5]:
    print(json.dumps(qa, ensure_ascii=False, indent=2))

print("\n" + "=" * 80)
print("Observation preview:")
obs = sample["observation"]
for key in list(obs.keys())[:2]:
    print("\n", key)
    print(type(obs[key]))
    print(json.dumps(obs[key], ensure_ascii=False, indent=2)[:1500])

print("\n" + "=" * 80)
print("Session summary preview:")
summaries = sample["session_summary"]
for key in list(summaries.keys())[:2]:
    print("\n", key)
    print(type(summaries[key]))
    print(json.dumps(summaries[key], ensure_ascii=False, indent=2)[:1500])