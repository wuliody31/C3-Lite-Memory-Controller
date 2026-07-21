from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from src.io_utils import read_json


class ProceduralRuleMatcher:
    def __init__(self, rules_path: Path | str):
        self.rules_path = Path(rules_path)
        self.rules: list[dict[str, Any]] = read_json(self.rules_path)

    def match(self, user_id: str, question: str, limit: int = 5) -> list[dict[str, Any]]:
        q = question.lower()
        tokens = [t for t in re.split(r'[^a-zA-Z0-9_]+', q) if len(t) >= 4]
        matched = []
        for rule in self.rules:
            if rule.get('user_id') != user_id:
                continue
            text = ' '.join([str(rule.get('name','')), str(rule.get('condition','')), str(rule.get('action',''))]).lower()
            lexical_hits = sorted({t for t in tokens if t in text})
            domain_trigger = any(x in q for x in ['project','supervisor','evidence','conflict','cv','interview','claim','travel','price','ticket','hotel','style','format'])
            score = len(lexical_hits) + int(rule.get('priority', 0))
            if lexical_hits or (domain_trigger and int(rule.get('priority', 0)) >= 5):
                matched.append({'id': rule['rule_id'], 'memory_type': 'procedural', 'text': f"{rule['condition']} -> {rule['action']}", 'name': rule['name'], 'priority': int(rule.get('priority',0)), 'score': score, 'matched_terms': lexical_hits})
        matched.sort(key=lambda x: x['score'], reverse=True)
        return matched[:limit]
