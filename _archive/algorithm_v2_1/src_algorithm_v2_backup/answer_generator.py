from __future__ import annotations
import json
from typing import Any


def format_evidence(selected_memories: list[dict[str, Any]], procedural_rules: list[dict[str, Any]]) -> str:
    lines = []
    for m in selected_memories:
        lines.append(f"[{m.get('id')}] [{m.get('memory_type')}] {m.get('text')}")
    for r in procedural_rules:
        lines.append(f"[{r.get('id')}] [procedural] {r.get('text')}")
    return '\n'.join(lines)


def generate_mock_answer(question: str, selected_memories: list[dict[str, Any]], conflict_result: dict[str, Any], procedural_rules: list[dict[str, Any]], query_type: str | None = None) -> str:
    if not selected_memories and not procedural_rules:
        return 'There is not enough stored evidence to answer this question.'
    lines = ['Answer based on selected memory evidence:']
    if query_type:
        lines.append(f'Query type: {query_type}')
    if selected_memories:
        lines.append('\nSelected evidence:')
        for m in selected_memories[:8]:
            lines.append(f"- [{m.get('id')}] [{m.get('memory_type')}] {m.get('text')}")
    if procedural_rules:
        lines.append('\nRelevant procedural rules:')
        for r in procedural_rules[:3]:
            lines.append(f"- [{r.get('id')}] {r.get('text')}")
    if conflict_result.get('outdated_memory_ids'):
        lines.append('\nConflict/update notes:')
        for note in conflict_result.get('conflict_notes', []):
            lines.append(f"- {note['older_memory']} is {note['relation']} by {note['newer_memory']}: {note['reason']}")
    lines.append('\nMemory explanation: the answer was generated from the selected episodic, semantic, and/or procedural evidence above.')
    return '\n'.join(lines)


def generate_openai_answer(question: str, selected_memories: list[dict[str, Any]], conflict_result: dict[str, Any], procedural_rules: list[dict[str, Any]], model: str, api_key: str | None) -> str:
    if not api_key:
        raise ValueError('OPENAI_API_KEY is missing.')
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    evidence_text = format_evidence(selected_memories, procedural_rules)
    conflict_text = json.dumps(conflict_result.get('conflict_notes', []), ensure_ascii=False, indent=2)
    prompt = f'''
You are a memory-grounded assistant for a C3-Lite multi-memory controller.

Question:
{question}

Selected memory evidence:
{evidence_text}

Conflict/update notes:
{conflict_text}

Instructions:
1. Answer only using the provided memory evidence.
2. If evidence is insufficient, say there is not enough stored evidence.
3. If older memory conflicts with newer memory, prefer the newer memory and mention the older memory only as historical context.
4. Briefly explain which memory types support your answer.
'''
    response = client.chat.completions.create(model=model, messages=[{'role':'system','content':'You are careful, evidence-grounded, and concise.'},{'role':'user','content':prompt}], temperature=0)
    return response.choices[0].message.content


def generate_answer(mode: str, question: str, selected_memories: list[dict[str, Any]], conflict_result: dict[str, Any], procedural_rules: list[dict[str, Any]], query_type: str | None = None, openai_model: str = 'gpt-4o-mini', openai_api_key: str | None = None) -> str:
    if mode == 'mock':
        return generate_mock_answer(question, selected_memories, conflict_result, procedural_rules, query_type)
    if mode == 'openai':
        return generate_openai_answer(question, selected_memories, conflict_result, procedural_rules, openai_model, openai_api_key)
    raise ValueError(f'Unknown answer mode: {mode}')
