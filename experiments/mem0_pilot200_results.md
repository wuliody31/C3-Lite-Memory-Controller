# Mem0 LoCoMo Pilot-200 Retrieval Results

## Experimental configuration

- Mem0 OSS: 2.0.15
- Memory insertion mode: `infer=False`
- Vector store: persistent local Qdrant
- Embedder: `multi-qa-MiniLM-L6-cos-v1`
- Embedding dimension: 384
- Embedding device: CPU
- Questions: 200
- Conversations: 10
- Stored memories: 8,423
  - Episodic: 5,882
  - Semantic: 2,541
- Source provenance: preserved
- Retrieval threshold: 0.0
- Reranking: disabled
- Slurm ingestion job: 36237
- Slurm query job: 36243

## Ingestion results

- Total ingestion time: 1,695.93 seconds
- Slurm elapsed time: 28 minutes 25 seconds
- Unique memory IDs: 8,423
- Empty conversations: 0
- Fatal errors: 0

## Retrieval results

| Configuration | Precision | Recall | Evidence F1 | Hit@K | Retrieved memories | Projected sources | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mem0 top-5 | 0.1217 | 0.4202 | 0.1805 | 0.4950 | 5.00 | 4.88 | 0.0430 s |
| Mem0 top-20 | 0.0453 | 0.5788 | 0.0819 | 0.6750 | 20.00 | 18.95 | 0.0434 s |

## Interpretation

Increasing the retrieval budget from five to twenty memories improved
mean evidence recall by 0.1586 and Hit@K by 0.1800. However, mean
precision decreased by 0.0763 and evidence F1 decreased by 0.0986.

The top-20 configuration failed to retrieve any gold-equivalent
evidence for 32.5% of the questions. Therefore, it is not sufficiently
reliable as a standalone replacement for the existing C3 retrieval
backend.

The result supports using Mem0 as a complementary candidate generator
whose outputs are filtered or fused by the C3 controller. It also
illustrates that increasing retrieval volume alone does not solve
memory selection and introduces substantial context noise.

## Scope limitation

This experiment evaluates a provenance-preserving Mem0 vector backend
with `infer=False`. It does not evaluate Mem0's LLM-based automatic
memory extraction, consolidation or lifecycle management and should
not be reported as the complete Mem0 system baseline.
