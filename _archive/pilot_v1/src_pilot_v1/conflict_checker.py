from __future__ import annotations
from typing import Any


def check_conflicts(adapter, memories: list[dict[str, Any]]) -> dict[str, Any]:
    resolved, outdated, conflict_notes, newer_ids = [], [], [], []
    for memory in memories:
        memory_id = memory.get('id')
        if not memory_id:
            continue
        superseding = adapter.find_superseding_memory(memory_id)
        if superseding:
            outdated.append(memory_id)
            for item in superseding:
                newer_ids.append(item['newer_memory'])
                conflict_notes.append({'older_memory': item['older_memory'], 'newer_memory': item['newer_memory'], 'relation': item['relation'], 'reason': item['reason']})
        else:
            resolved.append(memory)
    return {'resolved_memories': resolved, 'outdated_memory_ids': list(dict.fromkeys(outdated)), 'newer_memory_ids': list(dict.fromkeys(newer_ids)), 'conflict_notes': conflict_notes}


def enrich_with_newer_memories(adapter, conflict_result: dict[str, Any]) -> list[dict[str, Any]]:
    newer_memories = []
    for mid in conflict_result.get('newer_memory_ids', []):
        m = adapter.get_memory_by_id(mid)
        if m:
            newer_memories.append(m)
    return newer_memories
