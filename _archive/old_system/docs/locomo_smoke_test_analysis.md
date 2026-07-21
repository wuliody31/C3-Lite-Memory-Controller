\# LoCoMo One-Conversation Smoke Test Analysis



\## Purpose



This smoke test evaluates whether LoCoMo can be converted into the C3-Lite memory format, imported into Neo4j, and processed by the full evaluation pipeline.



\## Data



\- Sample: conv-26

\- Sessions: 19

\- Episodic memories: 419

\- Semantic facts: 184

\- Evaluation questions: 5



\## Main Result



The pipeline successfully ran all four methods: no memory, simple retrieval, all-memory, and C3-Lite controller.



C3-Lite achieved the highest evidence precision among retrieval-based methods while using substantially fewer memory items. Compared with all-memory, C3-Lite reduced the number of used memory items from 14.0 to 4.2, corresponding to approximately 70% context reduction.



\## Interpretation



The strict evidence recall remains low because LoCoMo gold evidence uses dialogue-turn IDs, while C3-Lite often retrieves semantic observations derived from those dialogue turns. Therefore, strict ID-level evidence scoring underestimates semantically valid retrieval in some cases.



\## Limitation



This is not a final LoCoMo performance result. It is an adapter validation and evidence-alignment diagnostic.

