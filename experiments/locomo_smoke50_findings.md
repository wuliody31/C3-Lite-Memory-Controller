# LoCoMo Qwen3 Smoke-50 Findings

## Execution

- Slurm job: 36158
- Model: Qwen/Qwen3-8B
- Quantisation: 4-bit NF4
- Questions: 50
- Methods: four
- Predictions: 200
- Empty answers: zero
- Execution state: completed with exit code 0

## Structural outcome

The LoCoMo adapter, namespaced dialogue identifiers, official
observation mapping, four-method execution and answer scorer all
completed successfully.

## Initial answer quality

Answer Token F1 over the 40 answerable questions was:

- simple retrieval: 0.1206
- all memory: 0.1157
- C3: 0.0793
- no memory: 0.0258

These values are descriptive smoke-test results and are not used for
statistical inference.

## Evidence audit

The strict evidence metric under-represented C3 because C3 usually
selected semantic observations whose source_ids pointed to the gold
episodic dialogue turns.

For C3:

- raw strict recall: 0.0300
- raw source-equivalent recall: 0.4245
- selected strict recall: 0.0100
- selected source-equivalent recall: 0.2683

For simple retrieval:

- selected strict recall: 0.4200
- selected source-equivalent recall: 0.4733

Source-equivalent evidence metrics are therefore required for the
LoCoMo evaluation.

## Controller behaviour

C3 routed 47 of 50 questions to semantic memory only, two to episodic
memory only and one to both episodic and semantic memory.

C3 selected an average of 2.04 memories, compared with 5.00 for simple
retrieval and 19.74 for all memory.

The smoke test indicates that the frozen controller compresses the
retrieved evidence substantially. This improves context efficiency but
also removes relevant source evidence on some external questions.

## Evaluation decision

No controller parameters will be changed using the smoke-test results.

The adapter and evaluation pipeline are sufficiently stable to proceed
to the frozen Pilot-200. Both strict and source-equivalent evidence
metrics will be retained.
