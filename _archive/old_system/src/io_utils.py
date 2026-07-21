from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def find_eval_questions_path(dataset_path: Path) -> Path:
    for p in [dataset_path/'data_eval'/'eval_questions.jsonl', dataset_path/'eval_questions.jsonl']:
        if p.exists():
            return p
    raise FileNotFoundError(f'Cannot find eval_questions.jsonl under {dataset_path}')


def find_procedural_rules_path(dataset_path: Path) -> Path:
    for p in [dataset_path/'data_raw'/'procedural_memory.json', dataset_path/'procedural_memory.json']:
        if p.exists():
            return p
    raise FileNotFoundError(f'Cannot find procedural_memory.json under {dataset_path}')
