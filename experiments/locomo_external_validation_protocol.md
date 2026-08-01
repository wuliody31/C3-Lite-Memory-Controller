# LoCoMo External Validation Protocol

## Purpose

This experiment evaluates whether the frozen C3 controller generalises
from the controlled Dataset A benchmark to an external long-term
conversation benchmark.

The experiment is external validation, not a new development set.
Controller thresholds, route rules, ranking weights, evidence-selection
logic and confidence thresholds must not be tuned using LoCoMo results.

## Source

- Dataset: official LoCoMo `locomo10.json`
- Source repository: snap-research/locomo
- Task used: question answering
- Conversation count: recorded from the official release
- Source repository commit and dataset SHA256 must be stored

## Memory mapping

### Episodic memory

Each dialogue turn becomes one episodic memory item.

Required fields:

- memory_id: original LoCoMo `dia_id`
- source_ids: original `dia_id`
- user_id: LoCoMo sample identifier
- timestamp: timestamp of the containing session
- text: speaker name and dialogue text
- session identifier
- speaker
- original conversation identifier

### Semantic memory

The official generated observation field may be used only when:

1. its structure is confirmed by schema inspection;
2. its source session can be traced;
3. all retrieval-based methods receive the same semantic items;
4. no additional C3-only LLM extraction is performed.

Semantic observations must not replace the original dialogue evidence.

### Procedural memory

LoCoMo does not provide user-specific procedural rules in the same form
as Dataset A. No artificial procedural memories will be added.

An empty procedural-memory file will be used unless the source data
contains an explicit instruction or policy field supported by the
official annotations.

## Question mapping

Each QA record becomes one evaluation question.

Preserved fields:

- sample identifier
- question identifier
- question
- reference answer
- LoCoMo category
- original evidence dialogue IDs
- speaker metadata where available

No synthetic route label will be added.

## Evaluation

Primary metrics:

- Answer Token F1 over all QA items
- Answer Token F1 by LoCoMo category

Retrieval metrics where gold evidence IDs are available:

- Evidence Precision
- Evidence Recall
- Evidence F1
- source-equivalent evidence overlap, only when its mapping is explicit

System metrics:

- average selected evidence count
- latency
- abstention rate
- empty-answer rate

Route F1 is not a primary LoCoMo metric because LoCoMo does not provide
gold episodic/semantic/procedural route annotations.

## Compared methods

- no_memory
- simple_retrieval
- all_memory
- c3

All methods use:

- the same Qwen3-8B backbone
- the same generation parameters
- the same source data
- the same evaluation questions
- the same answer scorer

## Stages

1. Schema audit
2. Adapter unit tests
3. Mock smoke test
4. Qwen3 smoke test
5. Frozen pilot
6. Formal full evaluation

## Anti-tuning policy

The smoke test may be used to repair:

- parsing errors
- broken ID mappings
- missing timestamps
- output-format errors
- scoring implementation errors

It must not be used to alter C3 controller behaviour.

The pilot may be used to estimate runtime and detect execution failures.
It must not be used to tune controller parameters.

Formal sample selection must be frozen before formal generation.
