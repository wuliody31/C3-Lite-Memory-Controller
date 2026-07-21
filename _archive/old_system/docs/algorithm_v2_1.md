# C3-Lite Algorithm v2.1

## Why v2.1 was introduced

The balanced v2 pilot improved route exact match, route F1 and evidence recall,
but evidence precision fell because the selected evidence set became wider.

A second evaluation issue was also identified: `question_type = abstention` does
not always imply that the correct action is to abstain. Some questions contain
explicit negative evidence and should be answered `No`.

Algorithm v2.1 therefore keeps the v2 route planner unchanged and focuses on:

1. temporal-aware shared ranking;
2. bidirectional graph relation expansion;
3. MMR evidence pruning;
4. answerability analysis;
5. selective abstention evaluation.

## Shared temporal-aware ranking

The three retrieval-based methods still share the same ranker.

Lexical relevance:

```text
L = 0.78 * BM25_norm + 0.22 * token_overlap
```

Episodic score:

```text
S_e = 0.66L + 0.16 importance + 0.18 temporal_prior
```

The temporal prior is query dependent:

```text
past-oriented query    -> prefer older events
current query          -> prefer recent events
change-over-time query -> neutral temporal prior; diversity is handled later
```

Semantic score:

```text
S_s = 0.70L + 0.14 confidence + 0.10 status_prior + 0.06 recency
```

Procedural score:

```text
S_p = 0.84L + 0.16 priority
```

## Bidirectional graph expansion

v2 only reliably expanded from an old memory to a newer related memory.

v2.1 queries MEMORY_RELATION edges touching either endpoint of the retrieved set.

For invalidating relations:

```text
SUPERSEDES
UPDATES
OVERRIDES
REPLACES
```

the resolver may pull both:

```text
newer/current memory
historical memory
```

For current-fact questions, historical evidence is excluded from usable evidence.

For episodic recall, temporal update, conflict resolution and explainability,
historical evidence can remain as historical context.

## MMR evidence selector

v2 filled the evidence budget by descending evidence score, which admitted
redundant memories.

v2.1 uses Maximal Marginal Relevance:

```text
MMR(m)
=
lambda * evidence_score(m)
-
(1-lambda) * max_similarity(m, selected)
```

with:

```text
lambda = 0.78
```

The evidence budget is adaptive to the number of selected routes and question type.

The selector also reserves:

- the strongest useful memory from each selected route;
- one historical memory for history-aware tasks when relevant.

## Answerability analysis

A factual verification question must distinguish direct proposition evidence from
uncertainty-policy evidence.

Example:

```text
model training --status--> not_required
```

is direct negative evidence and can answer:

```text
Did I decide to train my own model?
```

By contrast:

```text
production_deployment_claim --status--> unsupported_without_evidence
```

supports abstention for:

```text
Did I definitely deploy the system to production?
```

It does not prove deployment.

The answerability module classifies selected memories as:

```text
direct
uncertainty
policy
```

and produces:

```text
answerable
uncertain
insufficient
```

## Confidence calibration

The v2.1 adequacy score is:

```text
A =
0.30 * top_evidence
+ 0.15 * mean_top3
+ 0.15 * route_confidence
+ 0.20 * query_coverage
+ 0.10 * agreement
+ 0.10 * structural_support
```

The answerability state is evaluated before the adequacy thresholds.

```text
insufficient -> abstain
uncertain    -> caveat or abstain
answerable   -> direct / caveat based on adequacy
```

## Correct abstention ground truth

`question_type = abstention` is a capability category, not a binary gold action.

Preferred final benchmark field:

```json
"should_abstain": true
```

or:

```json
"should_abstain": false
```

For Dataset A v0.1, v2.1 falls back to:

```text
supporting_memory_ids is empty
-> should abstain
```

This fallback matches the current four unsupported factual claims in v0.1.

Before final held-out evaluation, the benchmark should contain an explicit,
manually reviewed `should_abstain` field.
