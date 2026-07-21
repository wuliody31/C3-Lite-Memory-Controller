from __future__ import annotations
import re

STOPWORDS = {'what','which','when','where','why','how','should','would','could','about','with','from','into','that','this','there','their','your','mine','current','currently','project','system','answer','question','explain','evidence','memory','user','does','have','need','want','我的','现在','当前','为什么','怎么','什么','是否','有没有'}
DOMAIN_TERMS = ['c3','lite','controller','scope','evaluation','baseline','dataset','episodic','semantic','procedural','conflict','temporal','abstention','explainability','supervisor','neo4j','rag','cv','interview','analyst','agent','hotel','travel','price','ticket','york']


def extract_keywords(question: str, max_keywords: int = 8) -> list[str]:
    q = question.lower()
    keywords = []
    for term in DOMAIN_TERMS:
        if term in q:
            keywords.append(term)
    for t in re.split(r'[^a-zA-Z0-9_]+', q):
        if len(t) >= 4 and t not in STOPWORDS and t not in keywords:
            keywords.append(t)
    if not keywords:
        keywords.append(q[:80])
    return keywords[:max_keywords]
