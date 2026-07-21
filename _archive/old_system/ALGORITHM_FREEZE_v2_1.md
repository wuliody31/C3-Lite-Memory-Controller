\# C3-Lite Algorithm v2.1 Freeze



Freeze date: 2026-07-05



Development dataset:

Dataset A v0.1



Balanced pilot:

12 questions



Frozen components:



\- Multi-signal query analyser

\- Context-adaptive route planner

\- Shared BM25 + metadata ranker

\- Temporal-aware episodic ranking

\- Relation-aware conflict resolver

\- Bidirectional memory-relation expansion

\- MMR evidence selector

\- Adaptive evidence budget

\- Answerability analyser

\- Confidence calibrator



Frozen parameters:



\- MMR lambda = 0.78

\- Episodic lexical weight = 0.66

\- Episodic importance weight = 0.16

\- Episodic temporal weight = 0.18

\- Semantic lexical weight = 0.70

\- Semantic confidence weight = 0.14

\- Semantic status weight = 0.10

\- Semantic recency weight = 0.06

\- Procedural lexical weight = 0.84

\- Procedural priority weight = 0.16

\- Direct threshold = 0.58

\- Caveat threshold = 0.45



Pilot acceptance result:



\- Route Exact Match = 0.750

\- Route F1 = 0.903

\- Evidence Recall = 0.724

\- Evidence Precision = 0.617

\- Average Retrieved Memories = 5.083

\- Average Used Memories = 4.250

\- Abstention Correctness = 1.000



No further parameter tuning will be performed using Dataset A v0.1 after the full development evaluation.

