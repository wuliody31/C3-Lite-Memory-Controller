# Mem0 LoCoMo Conversation-30 Retrieval Smoke

## Configuration

- Mem0 OSS: 2.0.15
- Python: 3.11.13
- Vector store: local Qdrant
- Embedder: multi-qa-MiniLM-L6-cos-v1
- Embedding dimension: 384
- Embedding device: CPU
- Memory insertion: infer=False
- Retrieval: dense top-5
- Conversation: locomo_conv_30
- Selection: deterministic category round-robin
- Questions: 10

## Structural results

- Ingested episodic memories: 369
- Unique source identifiers: 369
- Empty retrievals: 0
- Cross-conversation leakage: 0
- Persistent reopening: passed

## Retrieval results

| Metric | Result |
|---|---:|
| Hit@5 | 0.6000 |
| Mean Evidence Precision | 0.1200 |
| Mean Evidence Recall | 0.4333 |
| Mean Evidence F1 | 0.1821 |
| Mean retrieved memories | 5.00 |
| Mean retrieval latency | 0.016 seconds |

## Interpretation

The smoke experiment validates the Mem0 integration pipeline rather
than estimating formal retrieval performance. The result confirms
that raw LoCoMo turns can be stored with exact source provenance,
retrieved from persistent Qdrant storage and isolated by conversation.

No retrieval parameter was selected or modified based on these ten
questions.
