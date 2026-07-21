from __future__ import annotations

import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_'-]+|[\u4e00-\u9fff]+")


def normalise_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    tokens = [x.lower() for x in TOKEN_PATTERN.findall(text)]
    return [x for x in tokens if not stopwords or x not in stopwords]


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a and b else 0.0


def overlap_ratio(query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
    q, d = set(query_tokens), set(doc_tokens)
    return len(q & d) / len(q) if q else 0.0


def min_max_normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 if hi == 0 else 1.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def approximate_token_count(text: str) -> int:
    latin = len(re.findall(r"[A-Za-z0-9_'-]+", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return max(1, latin + cjk)


def split_clauses(query: str) -> list[str]:
    parts = re.split(r"\?|;|,|\band\b|\bthen\b|\bbut\b|\balso\b|以及|并且|然后|但是|，|；|？", query, flags=re.I)
    return [" ".join(x.strip().split()) for x in parts if len(x.strip()) >= 3]
