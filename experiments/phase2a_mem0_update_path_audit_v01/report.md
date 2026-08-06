# Phase II-A Mem0 Automatic Update-Path Audit

## Scope

This audit examines the frozen Mem0 OSS 2.0.15 source used in the
Phase II-A experiments.

- Source commit:
  `50bdaaea0c02744720ed374d88584fd01494eeb7`
- Automatic ingestion method:
  `Memory.add(..., infer=True)`

## Main finding

The automatic `infer=True` ingestion path is an additive memory-formation
pipeline. It uses `ADDITIVE_EXTRACTION_PROMPT`, whose declared sole
operation is ADD.

Existing memories are supplied to the LLM for duplicate suppression and
related-memory linking. Extracted facts are assigned new UUIDs, inserted
into the vector store, and recorded in history as ADD events.

## Update and delete boundary

The SDK implements public `update()` and `delete()` methods and their
corresponding internal helpers. These operations require explicit
external invocation. They are not called by the automatic
`add(..., infer=True)` pipeline.

Therefore, the presence of CRUD methods does not imply automatic
lifecycle state transition.

## Linked-memory boundary

The extraction prompt permits the LLM to emit `linked_memory_ids`.
However, the audited automatic persistence path does not use this field
to update, delete, supersede or archive existing memory records.

The entity store also contains a field named `linked_memory_ids`, but it
represents links from extracted entities to memory records. It is not a
semantic lifecycle relation such as SUPERSEDES.

## Experimental interpretation

The primary baseline should be described as:

**Mem0 OSS infer=True Additive Memory-Formation Baseline**

Validated capabilities:

- automatic fact extraction;
- exact-duplicate suppression;
- append-only persistence;
- history access;
- vector retrieval;
- explicit manual CRUD APIs.

Not provided by the automatic ingestion path:

- automatic semantic-state replacement;
- automatic supersession;
- stale-state deactivation;
- current-versus-historical arbitration;
- conflict-aware lifecycle policy.

## Consequence for C3

C3 Phase II-B should implement explicit automatic state transition over
the backend abstraction. A new semantic state can supersede an old
active state while preserving the transition as episodic memory.

This is a control-plane contribution rather than a replacement for
Mem0's storage or extraction capabilities.
