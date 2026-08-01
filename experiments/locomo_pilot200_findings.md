# LoCoMo Qwen3 Pilot-200 Findings

## Frozen execution

- Slurm job: 36163
- Model: Qwen/Qwen3-8B
- Quantisation: 4-bit NF4
- Questions: 200
- Answerable questions: 160
- Unanswerable questions: 40
- Methods: four
- Predictions: 800
- Execution state: completed with exit code 0
- Controller tuning after Pilot: none

## Answer quality

Answer Token F1 over answerable questions:

- simple retrieval: 0.1145
- all memory: 0.1116
- C3: 0.0875
- no memory: 0.0378

The Pilot confirms the Smoke-50 ordering. C3 improves substantially
over no memory but does not outperform the retrieval baselines on
lexical answer similarity.

## Source-equivalent evidence quality

Source-equivalent Evidence F1:

- C3: 0.2303
- simple retrieval: 0.1995
- all memory: 0.0808
- no memory: 0.0000

C3 achieved the highest source-equivalent Evidence F1 and precision
while selecting only 2.02 memories per question.

Source-equivalent evidence precision:

- C3: 0.1987
- simple retrieval: 0.1363
- all memory: 0.0467

Source-equivalent evidence recall:

- C3: 0.3104
- simple retrieval: 0.4425
- all memory: 0.5336

The controller therefore favours precision and context compression
over maximum recall.

## Efficiency

Mean selected evidence:

- C3: 2.02
- simple retrieval: 4.99
- all memory: 19.53

The results support the claim that C3 provides more selective and
compact memory context on an external long-conversation dataset.

## Abstention

C3 abstention accuracy was 0.795. It correctly abstained on only 2.5%
of unanswerable questions and incorrectly abstained on 1.25% of
answerable questions.

The confidence gate therefore did not generalise strongly to LoCoMo's
unanswerable category.

## Decision

The adapter, execution pipeline and source-equivalent evaluator are
stable enough for formal full-dataset evaluation.

No route rules, retrieval budgets, selector settings, confidence
thresholds or generation parameters will be modified after the Pilot.
