# C3-Lite Algorithm v2

## Why this version is stronger

The first pilot mixed two effects: controller logic and retrieval quality. Algorithm v2 makes the comparison fairer by giving **simple retrieval, all-memory and C3-Lite the same BM25 + metadata ranker**. C3-Lite differs by route selection, conflict resolution and confidence calibration.

## C1 — Context-adaptive route planning

The old first-match classifier is replaced by a multi-signal query profile. Signals for episodic recall, current facts, temporal change, conflict, procedural guidance, uncertainty and explanation are accumulated before routing.

Weighted route utility is logged, then structural gates prevent obvious over-routing:

```text
past-only recall          -> episodic
current fact              -> semantic
temporal update           -> episodic + semantic
general memory policy     -> procedural
domain-specific procedure -> procedural + semantic
decision conflict         -> episodic + semantic + procedural
```

## Shared retrieval algorithm

All retrieval-based methods share the same ranker.

Lexical relevance:

```text
L = 0.75 * normalised_BM25 + 0.25 * token_overlap
```

Episodic score:

```text
S_e = 0.68L + 0.17 importance + 0.15 recency
```

Semantic score:

```text
S_s = 0.68L + 0.14 confidence + 0.10 status_prior + 0.08 recency
```

Procedural score:

```text
S_p = 0.82L + 0.18 priority
```

A lexical gate prevents a recent or high-confidence but irrelevant memory from scoring highly.

## C2 — Conflict-aware resolution

Only invalidating relations mark the target as outdated:

```text
SUPERSEDES
UPDATES
OVERRIDES
REPLACES
```

Supportive relations do **not** invalidate the target:

```text
CLARIFIES
ALIGNS_WITH
DERIVED_FROM
```

This fixes a logical bug in the first pilot, where every `MEMORY_RELATION` was effectively treated as supersession.

The resolver also detects implicit semantic version conflicts:

```text
same subject + same relation + different objects
```

Preference order:

```text
active status > newer last_updated > higher confidence
```

For current-fact questions, outdated facts are excluded. For temporal, conflict and explanation questions, they can remain as `historical` evidence but cannot be treated as current truth.

## C3 — Confidence calibration

Evidence adequacy:

```text
A =
0.45 * top_evidence
+ 0.25 * mean_top3
+ 0.15 * route_confidence
+ 0.15 * agreement
```

The controller outputs one of:

```text
direct
caveat
abstain
```

A hard relevance gate is applied before the weighted thresholds.

## Fair baselines

### Simple retrieval

```text
shared ranker
fixed episodic + semantic routes
no procedural memory
no conflict resolution
no confidence calibration
```

### All-memory

```text
shared ranker
all three routes
no controlled routing
no conflict resolution
no confidence calibration
```

### C3-Lite

```text
adaptive routing
shared ranker
conflict resolution
confidence calibration
```

## Evaluation protocol

Use v0.1 as the **development/pilot benchmark**. Tune Algorithm v2 only on v0.1 and then freeze weights and thresholds.

For a stronger final claim, use the 24 newly added hard questions in Dataset A v0.2 as a held-out test subset. Do not tune Algorithm v2 on those 24 questions before reporting the held-out result.
