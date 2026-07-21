import json
from pathlib import Path


path = Path("../LoCoMo_C3/data_raw/locomo10.json")

if not path.exists():
    print("File not found:", path)
    print("Please put locomo10.json into ../LoCoMo_C3/data_raw/")
    raise SystemExit(1)

data = json.loads(path.read_text(encoding="utf-8"))

print("Top-level type:", type(data))

if isinstance(data, list):
    print("Number of samples:", len(data))
    sample = data[0]
elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    first_key = next(iter(data.keys()))
    sample = data[first_key]
    print("First key:", first_key)
else:
    raise TypeError(type(data))

print("\nSample type:", type(sample))

if isinstance(sample, dict):
    print("\nSample keys:")
    for key in sample.keys():
        print("-", key)

    print("\nPreview by key:")
    for key, value in sample.items():
        print("\nKEY:", key)
        print("TYPE:", type(value))

        if isinstance(value, list):
            print("LEN:", len(value))
            if value:
                print("FIRST ITEM TYPE:", type(value[0]))
                print("FIRST ITEM:", json.dumps(value[0], ensure_ascii=False)[:1000])

        elif isinstance(value, dict):
            print("KEYS:", list(value.keys())[:20])

        else:
            print(str(value)[:1000])
else:
    print("Sample preview:", str(sample)[:1000])